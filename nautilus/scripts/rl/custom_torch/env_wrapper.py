"""
maniskill — env factory for custom_torch train/eval/render.

Branches at runtime on cfg.gpu_sim:
  - True  → benchmark-native GPU-batched factory (IsaacGym / IsaacLab / ManiSkill).
            User fills in the factory call once; the placeholder raises a clear
            NotImplementedError until then.
  - False → gymnasium.vector.AsyncVectorEnv (or SyncVectorEnv) with a numpy↔torch
            wrapper so downstream algorithm code sees torch tensors on `cfg.device`.

Exposes:
    create_env(cfg)         — vec/parallel env for train + eval
    create_render_env(cfg)  — single env for render.py (1 env, render_mode='rgb_array')
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO     = Path(__file__).resolve().parents[4]   # actual repo root
NAUTILUS = Path(__file__).resolve().parents[3]   # <repo>/nautilus
sys.path.insert(0, str(NAUTILUS))
sys.path.insert(0, str(REPO))


# -----------------------------------------------------------------------------
# Per-benchmark single-env factory — auto-detects scripts/_<family>_env.py
# -----------------------------------------------------------------------------
def _make_single_env(task_id: str, render_mode=None, seed: int = 0):
    import gymnasium as gym
    env = None
    scripts_dir = REPO / "scripts"
    if scripts_dir.is_dir():
        for helper in scripts_dir.glob("_*_env.py"):
            # Some benchmarks (e.g. dm_control) check a mounted source dir into
            # the repo root that shadows the installed wheel. The helper module
            # is responsible for normalizing cwd, but we ALSO drop the repo root
            # from sys.path here so `from <pkg> import ...` inside the helper
            # finds the wheel, not the mount.
            _saved_path = list(sys.path)
            sys.path[:] = [p for p in sys.path if Path(p).resolve() != REPO]
            sys.path.insert(0, str(scripts_dir))
            try:
                module = __import__(helper.stem)
            except Exception:
                sys.path[:] = _saved_path
                continue
            for name in dir(module):
                if name.startswith("make_") and name.endswith("_env"):
                    fn = getattr(module, name)
                    try:
                        env = fn(task_id, render_mode=render_mode)
                    except TypeError:
                        env = fn(task_id)
                    break
            sys.path[:] = _saved_path
            break
    if env is None:
        env = gym.make(task_id, render_mode=render_mode)
    # The custom_torch algorithms (PPO/SAC/TD3 with MLP policies) require a
    # Box observation space. Wrap Dict / Tuple observations with FlattenObservation.
    if isinstance(env.observation_space, (gym.spaces.Dict, gym.spaces.Tuple)):
        env = gym.wrappers.FlattenObservation(env)
    return env


# -----------------------------------------------------------------------------
# CPU vec-env wrapper: numpy ↔ torch bridge over gymnasium.vector
# -----------------------------------------------------------------------------
class GymVecEnvWrapper:
    """Wraps a gymnasium.vector.VectorEnv so step/reset return torch tensors on `device`.

    Algorithm code (algo/sac.py etc.) assumes:
      env.step(action_torch)              -> (obs_torch, reward_torch, done_torch, info)
      env.reset()                         -> (obs_torch, extras) OR obs_torch
      env.observation_space.shape         -> shape of single env's obs (NOT including num_envs)
      env.action_space.shape[-1]          -> action dim
    """
    def __init__(self, vec_env, device: str | torch.device, episode_len: int | None = None):
        self.env = vec_env
        self.device = torch.device(device)
        self.num_envs = vec_env.num_envs
        # Expose single-env shapes (the algo code expects shape[0] == obs_dim, not num_envs).
        single_obs = vec_env.single_observation_space
        single_act = vec_env.single_action_space
        self._obs_shape = tuple(single_obs.shape)
        self._act_shape = tuple(single_act.shape)
        self.observation_space = type("_Spc", (), {"shape": self._obs_shape})()
        self.action_space = type("_Spc", (), {"shape": self._act_shape})()
        self.max_episode_length = episode_len

    def _to_torch(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(np.asarray(x)).float().to(self.device)
        if isinstance(x, (list, tuple)):
            return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple):
            obs, info = out[0], out[1]
        else:
            obs, info = out, {}
        return self._to_torch(obs), info

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action_np = action.detach().cpu().numpy()
        else:
            action_np = np.asarray(action)
        # Clip to bounds for envs with bounded action_space (most continuous envs).
        try:
            low = self.env.single_action_space.low
            high = self.env.single_action_space.high
            action_np = np.clip(action_np, low, high)
        except Exception:
            pass
        step_out = self.env.step(action_np)
        if len(step_out) == 5:
            obs, reward, term, trunc, info = step_out
            done = np.logical_or(term, trunc)
            if isinstance(info, dict):
                info = {**info, "TimeLimit.truncated": self._to_torch(trunc).bool()}
        else:
            obs, reward, done, info = step_out
        return (self._to_torch(obs),
                self._to_torch(reward),
                self._to_torch(done).long(),
                info if isinstance(info, dict) else {})


def _make_cpu_vec_env(task_id: str, num_envs: int, seed: int):
    import gymnasium as gym
    factories = []
    for i in range(num_envs):
        s = seed + i
        def _f(task_id=task_id, s=s):
            env = _make_single_env(task_id, render_mode=None, seed=s)
            return env
        factories.append(_f)
    if num_envs == 1:
        return gym.vector.SyncVectorEnv(factories)
    return gym.vector.AsyncVectorEnv(factories)


# -----------------------------------------------------------------------------
# Public factories
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# ManiSkill GPU-batched wrapper — env returns torch tensors with leading num_envs.
#   - observation_space.shape on the raw env is (num_envs, obs_dim); MLP code
#     expects per-env shape, so we expose (obs_dim,) via a simple shim object.
#   - step returns (obs, reward, term, trunc, info) as torch tensors on cuda.
#   - info["success"] is a torch bool tensor shaped (num_envs,).
# -----------------------------------------------------------------------------
def _stackcube_v1_decompose(env, reward, info):
    """Mask-partitioned sum decomposition of StackCube-v1's staged reward.

    Source: mani_skill/envs/tasks/tabletop/stack_cube.py::compute_dense_reward.
    Stages are mutually exclusive (success > placed > grasped > reach) so
    Σ terms == reward exactly per env per step.
    """
    base = env
    while hasattr(base, "env") and not hasattr(base, "agent"):
        base = base.env
    base = getattr(base, "unwrapped", base)
    tcp_pose = base.agent.tcp.pose.p
    cubeA_pos = base.cubeA.pose.p
    cubeB_pos = base.cubeB.pose.p
    half = base.cube_half_size

    d_tcp = torch.linalg.norm(tcp_pose - cubeA_pos, dim=1)
    base_reach = 2.0 * (1.0 - torch.tanh(5.0 * d_tcp))

    goal_xyz = torch.hstack(
        [cubeB_pos[:, 0:2], (cubeB_pos[:, 2] + half[2] * 2)[:, None]]
    )
    d_goal = torch.linalg.norm(goal_xyz - cubeA_pos, dim=1)
    place_reward = 1.0 - torch.tanh(5.0 * d_goal)

    gripper_width = (base.agent.robot.get_qlimits()[0, -1, 1] * 2).to(base.device)
    ungrasp_reward = torch.sum(base.agent.robot.get_qpos()[:, -2:], dim=1) / gripper_width
    grasped_now = info.get("is_cubeA_grasped")
    if grasped_now is None:
        grasped_now = base.agent.is_grasping(base.cubeA)
    ungrasp_reward = torch.where(grasped_now, ungrasp_reward, torch.ones_like(ungrasp_reward))
    v = torch.linalg.norm(base.cubeA.linear_velocity, dim=1)
    av = torch.linalg.norm(base.cubeA.angular_velocity, dim=1)
    static_reward = 1.0 - torch.tanh(v * 10.0 + av)

    success = info["success"].bool()
    on_cubeB = info["is_cubeA_on_cubeB"].bool()
    grasped = info["is_cubeA_grasped"].bool()
    m_success = success
    m_place   = on_cubeB & ~success
    m_grasp   = grasped & ~on_cubeB & ~success
    m_reach   = ~grasped & ~on_cubeB & ~success

    t_reach   = m_reach.float()   * base_reach
    t_grasp   = m_grasp.float()   * (4.0 + place_reward)
    t_place   = m_place.float()   * (6.0 + (ungrasp_reward + static_reward) / 2.0)
    t_success = m_success.float() * 8.0
    return {
        "reach":   t_reach,
        "grasp":   t_grasp,
        "place":   t_place,
        "success": t_success,
    }


_MANISKILL_DECOMPOSERS = {
    "StackCube-v1": _stackcube_v1_decompose,
}


class ManiSkillVecEnvWrapper:
    def __init__(self, env, device, num_envs: int, episode_len: int | None = None,
                 decomposer=None, task_id: str | None = None,
                 invariant_atol: float = 1e-4, invariant_rtol: float = 1e-3):
        self.env = env
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        # Strip the leading num_envs dim from the spaces' shapes for downstream
        # algo code (MLPNet picks shape[0] when given a Sequence).
        raw_obs_shape = tuple(env.observation_space.shape)
        raw_act_shape = tuple(env.action_space.shape)
        self._obs_shape = raw_obs_shape[1:] if len(raw_obs_shape) > 1 else raw_obs_shape
        self._act_shape = raw_act_shape[1:] if len(raw_act_shape) > 1 else raw_act_shape
        self.observation_space = type("_Spc", (), {"shape": self._obs_shape})()
        self.action_space = type("_Spc", (), {"shape": self._act_shape})()
        self.max_episode_length = episode_len
        self.decomposer = decomposer
        self.task_id = task_id
        self._invariant_atol = float(invariant_atol)
        self._invariant_rtol = float(invariant_rtol)
        self._invariant_checked = False

    def _t(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(self.device)
        return torch.as_tensor(x, device=self.device)

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple):
            obs, info = out[0], out[1]
        else:
            obs, info = out, {}
        return self._t(obs).float(), info

    def step(self, action):
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, device=self.device, dtype=torch.float32)
        else:
            action = action.to(self.device).float()
        out = self.env.step(action)
        if len(out) == 5:
            obs, reward, term, trunc, info = out
            term_t = self._t(term).bool()
            trunc_t = self._t(trunc).bool()
            done = torch.logical_or(term_t, trunc_t).long()
            if isinstance(info, dict):
                # ManiSkillVectorEnv auto-resets done envs in-place and
                # replaces top-level `info` with the post-reset dict,
                # stashing the pre-reset step info under `info["final_info"]`.
                # If the env populated `detailed_reward` inside
                # `compute_dense_reward`, it now lives in final_info — lift
                # it back so the per-term trackers (algo/ac_base.py) see it
                # on the done step.
                if "detailed_reward" not in info and isinstance(info.get("final_info"), dict):
                    fi = info["final_info"]
                    if "detailed_reward" in fi:
                        info = {**info, "detailed_reward": fi["detailed_reward"]}
                info = {**info, "TimeLimit.truncated": trunc_t}
        else:
            obs, reward, done, info = out
            done = self._t(done).long()
        if self.decomposer is not None and isinstance(info, dict):
            reward_t = self._t(reward).float()
            terms = self.decomposer(self.env, reward_t, info)
            # Invariant: only check on non-reset steps. ManiSkillVectorEnv
            # auto-resets done envs in-place, so on done steps the physics
            # state we read post-step is the fresh reset state, while
            # `reward` was computed from the pre-reset state — a small drift
            # is unavoidable at the horizon boundary.
            if not self._invariant_checked and not bool(done.bool().any()):
                total = sum(terms.values())
                if not torch.allclose(total, reward_t,
                                      atol=self._invariant_atol,
                                      rtol=self._invariant_rtol):
                    diff = (total - reward_t).abs().max().item()
                    raise RuntimeError(
                        f"[detailed_reward] {self.task_id}: Σ terms != reward "
                        f"(max|Δ|={diff:.3e}, atol={self._invariant_atol})"
                    )
                self._invariant_checked = True
            info = {**info, "detailed_reward": terms, "reward_composer": "sum"}
        obs_t = self._t(obs).float()
        reward_t = self._t(reward).float()
        # Defensive NaN/Inf guards. If a SAPIEN GPU race or a pose-composition
        # edge case ever produces a non-finite obs/reward, replace with 0 and
        # log a one-time warning so the rest of the rollout doesn't poison
        # the policy gradient.
        if not torch.isfinite(obs_t).all() or not torch.isfinite(reward_t).all():
            if not getattr(self, "_nonfinite_warned", False):
                n_obs_bad = int((~torch.isfinite(obs_t)).any(dim=-1).sum())
                n_rew_bad = int((~torch.isfinite(reward_t)).sum())
                print(f"[env_wrapper WARN] non-finite obs/reward: "
                      f"n_obs_envs={n_obs_bad}, n_rew_envs={n_rew_bad}", flush=True)
                self._nonfinite_warned = True
            obs_t = torch.nan_to_num(obs_t, nan=0.0, posinf=0.0, neginf=0.0)
            reward_t = torch.nan_to_num(reward_t, nan=0.0, posinf=0.0, neginf=0.0)
        return (obs_t,
                reward_t,
                done,
                info if isinstance(info, dict) else {})

    def render(self):
        return self.env.render()

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


def _make_maniskill_gpu_env(task: str, num_envs: int, seed: int, device: str,
                             render_mode=None, control_mode=None,
                             ignore_terminations: bool = True):
    """Build a GPU-batched ManiSkill env and wrap with ManiSkillVectorEnv.

    The raw `gym.make(...)` does NOT auto-reset done envs — once an env hits
    truncation/termination, subsequent `step()` calls keep returning done=True,
    which collapses training-time episode_length to 1 after the first rollout.
    ManiSkill's canonical training pattern (mani_skill/examples/baselines/sac/sac.py)
    wraps with `ManiSkillVectorEnv(..., ignore_terminations=True, record_metrics=True)`
    so all envs reset together at the horizon (clean GAE boundaries for PPO).

    For tasks whose IsaacLab counterpart uses `time_out=False` on a success
    DoneTerm (i.e. success TERMINATES the episode and the value bootstrap is
    zero), pass `ignore_terminations=False` so ManiSkillVectorEnv honors the
    env's `info["success"]` -> terminated flag and the reward integral matches
    the IsaacLab apples-to-apples baseline.
    """
    import mani_skill.envs  # noqa: F401 - registers @register_env tasks
    import gymnasium as gym
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
    env = gym.make(
        task,
        obs_mode="state",
        reward_mode="dense",
        control_mode=control_mode,
        render_mode=render_mode,
        num_envs=int(num_envs),
        sim_backend="auto",
    )
    env = ManiSkillVectorEnv(
        env,
        num_envs=int(num_envs),
        ignore_terminations=ignore_terminations,
        record_metrics=True,
    )
    return env


def create_env(cfg):
    task = str(cfg.get("task"))
    n = int(cfg.get("num_envs", cfg.get("n_envs", 1)))
    seed = int(cfg.get("seed", 0))
    device = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    gpu_sim = bool(cfg.get("gpu_sim", False))

    if gpu_sim:
        # ManiSkill 3 — state-mode, GPU-batched. The env returns torch tensors
        # of shape (num_envs, ...) directly, so we use a thin wrapper that
        # exposes per-env observation/action shapes for the MLP policies.
        control_mode = cfg.get("control_mode", None)
        ignore_terminations = bool(cfg.get("ignore_terminations", True))
        env = _make_maniskill_gpu_env(
            task, num_envs=n, seed=seed, device=device,
            render_mode=None, control_mode=control_mode,
            ignore_terminations=ignore_terminations,
        )
        try:
            env.reset(seed=seed)
        except TypeError:
            env.reset()
        decomposer = _MANISKILL_DECOMPOSERS.get(task)
        return ManiSkillVecEnvWrapper(env, device=device, num_envs=n,
                                      decomposer=decomposer, task_id=task)
    vec = _make_cpu_vec_env(task, n, seed)
    return GymVecEnvWrapper(vec, device=device)


def create_render_env(cfg):
    """Single env (n=1) with rendering enabled — used by render.py."""
    task = str(cfg.get("task"))
    seed = int(cfg.get("seed", 0))
    device = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    gpu_sim = bool(cfg.get("gpu_sim", False))

    if gpu_sim:
        # ManiSkill 3 render env — single GPU env with rgb_array render_mode.
        control_mode = cfg.get("control_mode", None)
        env = _make_maniskill_gpu_env(task, num_envs=1, seed=seed, device=device,
                                      render_mode="rgb_array", control_mode=control_mode)
        try:
            env.reset(seed=seed)
        except TypeError:
            env.reset()
        wrapped = ManiSkillVecEnvWrapper(env, device=device, num_envs=1)
        wrapped._raw_env = env
        return wrapped

    env = _make_single_env(task, render_mode="rgb_array", seed=seed)
    # Wrap in a 1-env vector so the agent code path stays uniform.
    import gymnasium as gym

    class _SingleVec:
        num_envs = 1
        single_observation_space = env.observation_space
        single_action_space = env.action_space

        def reset(self, **kw):
            o = env.reset(**kw)
            obs = o[0] if isinstance(o, tuple) else o
            return np.expand_dims(np.asarray(obs), 0), (o[1] if isinstance(o, tuple) else {})

        def step(self, action_np):
            a = action_np[0] if action_np.ndim > 1 else action_np
            out = env.step(a)
            if len(out) == 5:
                obs, r, term, trunc, info = out
                return (np.expand_dims(obs, 0), np.array([r]),
                        np.array([term]), np.array([trunc]), info)
            obs, r, done, info = out
            return np.expand_dims(obs, 0), np.array([r]), np.array([done]), info

        def render(self):
            return env.render()

    wrapped = GymVecEnvWrapper(_SingleVec(), device=device)
    wrapped._raw_env = env  # render.py uses this for frame extraction
    return wrapped
