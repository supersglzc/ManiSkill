"""
maniskill — render a trained policy to video.mp4 (custom_torch).

Single env (n=1), offscreen rendering enabled, runs ONE episode with the loaded
checkpoint and writes:

    <checkpoint-dir>/render.mp4
    <checkpoint-dir>/render_contact_sheet.jpg

Always produces a visible artifact — works on headless hosts because it uses
offscreen rendering. To open a live window, use the env's native viewer.

Example:
    python nautilus/scripts/rl/custom_torch/render.py algo=ppo task=StackCube-v1 \\
        checkpoint=nautilus/rl_experiments/runs/<trial_id>/checkpoint.pth
"""
from __future__ import annotations

import sys
from pathlib import Path

# IsaacGym/bidexhands quirk: pre-import any `<repo>/scripts/_*_env.py` helper
# BEFORE `import torch`, since some sims (IsaacGym) refuse to load if torch is
# already imported. No-op for repos without such a helper.
_REPO_FOR_PREIMPORT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_FOR_PREIMPORT / "scripts"))
for _hp in (_REPO_FOR_PREIMPORT / "scripts").glob("_*_env.py"):
    try:
        __import__(_hp.stem)  # noqa: F401
        break
    except ImportError:
        continue

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

REPO     = Path(__file__).resolve().parents[4]   # actual repo root
NAUTILUS = Path(__file__).resolve().parents[3]   # <repo>/nautilus
sys.path.insert(0, str(NAUTILUS))
sys.path.insert(0, str(REPO))

CUSTOM_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(CUSTOM_ROOT))

from algo import alg_name_to_path  # noqa: E402
from utils.common import load_class_from_path  # noqa: E402
from env_wrapper import create_render_env  # noqa: E402


def _resolve_algo_class(name: str):
    target = "Agent" + name.upper() if not name.startswith("Agent") else name
    if target in alg_name_to_path:
        return load_class_from_path(target, alg_name_to_path[target])
    for cls_name, path in alg_name_to_path.items():
        if cls_name.lower() == target.lower():
            return load_class_from_path(cls_name, path)
    raise KeyError(f"Algorithm '{name}' not found in registry.")


def _grab_frame(env) -> np.ndarray | None:
    raw = getattr(env, "_raw_env", None)
    target = raw if raw is not None else env
    try:
        frame = target.render()
    except Exception:
        return None
    # torch tensor from GPU-batched sims (ManiSkill returns (N, H, W, 3) uint8)
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    if isinstance(frame, np.ndarray):
        if frame.ndim == 4 and frame.shape[0] == 1:
            frame = frame[0]
        return frame.astype(np.uint8)
    if isinstance(frame, list) and frame and isinstance(frame[0], np.ndarray):
        return frame[0].astype(np.uint8)
    return None


def _write_mp4(frames: list[np.ndarray], path: Path, fps: int = 30) -> None:
    if not frames:
        print("[render] no frames captured — skipping mp4 write", file=sys.stderr); return
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio  # type: ignore[no-redef]
    imageio.mimsave(str(path), frames, fps=fps, codec="libx264", quality=8)


def _write_contact_sheet(frames: list[np.ndarray], path: Path) -> None:
    if not frames:
        return
    try:
        from PIL import Image
    except ImportError:
        return
    n = len(frames)
    indices = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})
    sample = [Image.fromarray(frames[i]) for i in indices]
    h = max(im.height for im in sample)
    w = sum(im.width for im in sample)
    sheet = Image.new("RGB", (w, h))
    x = 0
    for im in sample:
        sheet.paste(im, (x, 0))
        x += im.width
    sheet.save(str(path), quality=85)


@hydra.main(version_base=None, config_path="../../../configs/rl", config_name="ppo")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    if "algo" not in cfg or not isinstance(cfg.get("algo"), DictConfig):
        algo_name = str(cfg.get("algo", "ppo"))
        cfg.algo = OmegaConf.create({
            "name": algo_name,
            "act_class": cfg.get("act_class") or _default_act_class(algo_name),
            "cri_class": cfg.get("cri_class") or _default_cri_class(algo_name),
            "actor_lr": 3.0e-4, "critic_lr": 3.0e-4,
            "tracker_len": 100, "obs_norm": False,
            "alpha": cfg.get("alpha", None), "alpha_lr": 3.0e-4,
            "no_tgt_actor": True, "nstep": 1,
        })
    cfg.device = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    cfg.num_envs = 1
    # Honor cfg.gpu_sim — GPU-batched simulators (IsaacLab / IsaacGym / ManiSkill GPU)
    # cannot be replaced with a CPU gymnasium.make for render. create_render_env knows
    # how to launch with cameras enabled when gpu_sim=True.

    task = cfg.get("task")
    ckpt_path = cfg.get("checkpoint")
    fps = int(cfg.get("fps", 30))
    max_steps = int(cfg.get("max_steps", 1000))
    if task is None:
        print("[error] task=<id> is required.", file=sys.stderr); sys.exit(2)
    if not ckpt_path:
        print("[error] checkpoint=<path> is required.", file=sys.stderr); sys.exit(2)
    ckpt = Path(ckpt_path)
    if not ckpt.is_absolute():
        ckpt = REPO / ckpt
    if not ckpt.exists():
        print(f"[error] checkpoint not found: {ckpt}", file=sys.stderr); sys.exit(2)

    env = create_render_env(cfg)
    # Tighten max_steps to the env's episode horizon when available — avoids
    # rendering 1000 steps on a 300-step env.
    raw_env = getattr(env, "_raw_env", env)
    env_max_len = getattr(getattr(raw_env, "unwrapped", raw_env), "max_episode_length", None)
    if env_max_len is not None and int(env_max_len) > 0:
        max_steps = min(max_steps, int(env_max_len) + 1)
        print(f"[render] capping max_steps at env.max_episode_length+1 = {max_steps}", flush=True)
    # Pre-load checkpoint to detect whether obs/value normalization was used
    # during training. If yes, force-enable on cfg.algo BEFORE building the
    # agent so the agent allocates obs_rms / value_rms ready to be populated.
    # Without this, the actor (trained on normalized obs) sees raw OOD obs at
    # inference and produces garbage — symptom is render returns far below
    # the training-time episodic_return_mean.
    state = torch.load(str(ckpt), map_location=cfg.device, weights_only=False)
    if state.get("obs_norm") or "obs_rms" in state:
        OmegaConf.update(cfg, "algo.obs_norm", True, merge=True)
    if state.get("value_norm") or "value_rms" in state:
        OmegaConf.update(cfg, "algo.value_norm", True, merge=True)
    agent_cls = _resolve_algo_class(str(cfg.algo.name))
    agent = agent_cls(env=env, cfg=cfg)
    agent.actor.load_state_dict(state["actor"])
    agent.critic.load_state_dict(state["critic"])
    if "obs_rms" in state and getattr(agent, "obs_rms", None) is not None:
        agent.obs_rms.load_state_dict(state["obs_rms"])
        print("[render] restored obs_rms from checkpoint", flush=True)
    if "value_rms" in state and getattr(agent, "value_rms", None) is not None:
        agent.value_rms.load_state_dict(state["value_rms"])
        print("[render] restored value_rms from checkpoint", flush=True)
    agent.actor.eval()

    frames: list[np.ndarray] = []
    obs, _ = env.reset()

    # Position the IsaacLab viewer camera for a wide isometric view that shows
    # the full robot + table + workspace. Default eye=(2.5, 2.5, 1.6) m diagonal
    # from workspace center, looking slightly down at target=(0.30, 0, 0.40).
    # Override per-task via cfg.viewer_eye / cfg.viewer_target.
    try:
        viewer_eye    = list(cfg.get("viewer_eye",    [2.5, 2.5, 1.6]))
        viewer_target = list(cfg.get("viewer_target", [0.30, 0.0, 0.4]))
        raw_env = getattr(env, "_raw_env", env)
        env_origin = raw_env.unwrapped.scene.env_origins[0].cpu().numpy().tolist()
        eye    = [viewer_eye[i]    + env_origin[i] for i in range(3)]
        target = [viewer_target[i] + env_origin[i] for i in range(3)]
        raw_env.unwrapped.sim.set_camera_view(eye=eye, target=target)
        print(f"[render] viewer camera   eye={eye}  target={target}", flush=True)
    except Exception as _camera_err:
        print(f"[render] could not set viewer camera ({_camera_err}) — using default", flush=True)

    ep_return = 0.0
    for step in range(max_steps):
        f = _grab_frame(env)
        if f is not None:
            frames.append(f)
        with torch.no_grad():
            # Normalize obs via obs_rms BEFORE calling the actor — training's
            # rollout path (agent.get_actions in ppo.py) normalizes first, so
            # the actor was trained on normalized inputs. Skipping this step
            # silently produces near-zero performance even with the correct
            # checkpoint loaded.
            obs_in = agent.obs_rms.normalize(obs) if getattr(agent, "obs_rms", None) is not None else obs
            if hasattr(agent.actor, "get_actions"):
                # Stochastic eval — sample from the action distribution rather than
                # take the deterministic mean. For under-trained PPO policies the
                # mean is near-zero (entropy hasn't collapsed yet) and the robot
                # stays stationary, producing 301 visually-identical frames.
                action = agent.actor.get_actions(obs_in, sample=True)
                if isinstance(action, tuple):
                    action = action[0]
            else:
                action = agent.actor(obs_in)
        next_obs, reward, done, info = env.step(action)
        if isinstance(reward, torch.Tensor):
            ep_return += float(reward.mean().item())   # mean across envs (works for n=1 too)
        else:
            ep_return += float(reward)
        # Multi-env render: if num_envs > 1 the env auto-resets terminated envs
        # internally — we should NOT break on first done, just keep rendering for
        # max_steps. Single-env (.numel() == 1) IS the episode terminator.
        if isinstance(done, torch.Tensor) and done.numel() > 1:
            obs = next_obs; continue
        if bool(done.item() if isinstance(done, torch.Tensor) else done):
            f = _grab_frame(env)
            if f is not None:
                frames.append(f)
            break
        obs = next_obs

    out_dir = ckpt.parent
    mp4 = out_dir / "render.mp4"
    sheet = out_dir / "render_contact_sheet.jpg"
    _write_mp4(frames, mp4, fps=fps)
    _write_contact_sheet(frames, sheet)
    print(f"[render] wrote {mp4} ({len(frames)} frames, return={ep_return:.4f})", flush=True)

    # Hard-exit BEFORE Kit / sim shutdown. With some IsaacLab + Isaac Sim
    # combos (e.g. 5.1) `sim_app.close()` hangs on USD stage detach — render
    # hangs indefinitely AFTER the MP4 is on disk. The MP4 + contact sheet
    # are flushed above, so we skip the cleanup. Avoids an external watchdog.
    import os as _os
    _os.sync()
    _os._exit(0)


def _default_act_class(algo: str) -> str:
    a = algo.lower()
    if a == "ppo": return "DiagGaussianMLPPolicy"
    if a == "sac": return "TanhDiagGaussianMLPPolicy"
    return "TanhMLPPolicy"


def _default_cri_class(algo: str) -> str:
    return "MLPCritic" if algo.lower() == "ppo" else "DoubleQ"


if __name__ == "__main__":
    main()
