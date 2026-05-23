"""StackCube-Tower-v1 — 3-cube tower stacking, faithfully mirrors IsaacLab Triton-Franka-StackCube.

Maps the IsaacLab manager-based env at
``source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack_cube/`` into
the ManiSkill conventions. Three identical cubes (edge length CUBE_SIZE = 0.043 m)
named ``cube_0`` / ``cube_1`` / ``cube_2`` — tower bottom-up: ``cube_2`` (base on
table) <- ``cube_1`` (middle) <- ``cube_0`` (top).

Action (4-D ``Box(-1, 1, (4,), float32)``): the IsaacLab task uses
``EMACumulativeDeltaPositionAction`` with **scale=0.01, alpha=0.5** (xyz only,
RPY locked) + binary gripper. We mirror this with ManiSkill
``pd_ee_target_delta_pos`` (cumulative-target IK delta) and an explicit
**0.1× pre-scale on the xyz channels in ``step()``**. Combined with the
default ``pos_upper=0.1`` of ``pd_ee_target_delta_pos``, that gives an
effective ``±0.01 m`` per-step delta (the IsaacLab scale) sent to the IK
controller. EMA smoothing (alpha) is omitted — secondary effect.

Robot base pose (world frame): ``(-0.274, 0.49, 0.01)`` — matches the
``bidex StackCube``-derived IsaacLab base placement so the robot-relative
cube geometry is identical.

Robot initial joint pose (mirrors IsaacLab ``FRANKA_INIT_JOINT_POS``):
``joint1=-0.785, joint2=-0.785, joint3=0.0, joint4=-2.655, joint5=0.0,
joint6=1.87, joint7=0.0, fingers=0.04``. Gaussian noise ``±0.02`` rad
applied at reset.

Termination: time-out only. IsaacLab pins ``termination_on_success = False``;
ManiSkill's ``BaseEnv.step()`` reads ``info["success"]`` and writes
``terminated`` automatically, so we override ``step()`` to force
``terminated`` to all-False. ``TimeLimitWrapper`` still truncates at
``max_episode_steps=180`` (= 9 s at 20 Hz).

Observation (19-dim ``Box(-inf, inf, (1, 19), float32)``, ``obs_mode="state"``,
**exact mirror of IsaacLab's ``ObservationsCfg.PolicyCfg`` term order**):

    ee_pose                  (7,)  — TCP position(3) + quaternion(4) in robot root frame
    grasping_cube_position   (3,)  — mux on _cube_0_on_cube_1_predicate(env) (robot root frame)
    grasping_target_position (3,)  — mux on _cube_0_on_cube_1_predicate(env) (robot root frame)
    gripper_pos              (2,)  — qpos of the 2 finger joints
    last_action              (4,)  — previous policy action (3 xyz delta + 1 gripper)

Mux rule (stateless per-step):

    predicate False (cube_0 NOT yet on cube_1):
        grasping_cube   <- cube_0
        grasping_target <- cube_1.xyz + [0, 0, CUBE_SIZE]
    predicate True  (cube_0 ON cube_1):
        grasping_cube   <- cube_2
        grasping_target <- cube_0.xyz + [0, 0, CUBE_SIZE]

The §6 reward reuses ``_cube_0_on_cube_1_predicate`` from this module.
"""

from typing import Any, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


# Cube edge length + mass — must stay in sync with the constants used by §6 / §7
# (mirrors the values in IsaacLab's
# stack_cube/config/franka/joint_pos_env_cfg.py:CUBE_SIZE / CUBE_MASS).
CUBE_SIZE = 0.043
CUBE_HALF_SIZE = CUBE_SIZE / 2.0
CUBE_MASS = 0.055


def _cube_pos_in_robot_root_frame(env: "StackCubeTowerEnv", cube_key: str) -> torch.Tensor:
    """xyz of ``env.<cube_key>`` expressed in the robot root frame (B, 3)."""
    cube = getattr(env, cube_key)
    root_pose_inv = env.agent.robot.pose.inv()
    # Pose * Pose composition handles the rotate-then-translate for us; we only
    # need the position component.
    return (root_pose_inv * cube.pose).p


def _stack_target_in_robot_root_frame(env: "StackCubeTowerEnv", base_cube_key: str) -> torch.Tensor:
    """``base_cube.pos_w + [0, 0, CUBE_SIZE]`` expressed in the robot root frame (B, 3).

    Builds a synthetic Pose at the target world location, then transforms it
    into the robot root frame the same way ``_cube_pos_in_robot_root_frame``
    does. Quaternion is identity (target is position-only).
    """
    base = getattr(env, base_cube_key)
    target_pos_w = base.pose.p.clone()
    target_pos_w[:, 2] = target_pos_w[:, 2] + CUBE_SIZE
    quat = torch.zeros((target_pos_w.shape[0], 4), device=target_pos_w.device)
    quat[:, 0] = 1.0
    target_pose_w = Pose.create_from_pq(p=target_pos_w, q=quat)
    return (env.agent.robot.pose.inv() * target_pose_w).p


def _cube_0_on_cube_1_predicate(
    env: "StackCubeTowerEnv",
    xy_threshold: float = 0.02,
    z_threshold: float = 0.01,
) -> torch.Tensor:
    """(B,) bool tensor — True per env when ``cube_0`` is stacked on ``cube_1``.

    Mirrors the IsaacLab predicate of the same name in
    ``stack_cube/mdp/observations.py``. §5 obs and §6 reward must share this
    helper to keep the state-machine in lock-step.
    """
    top = env.cube_0.pose.p
    bot = env.cube_1.pose.p
    xy_dist = torch.linalg.norm(top[:, :2] - bot[:, :2], dim=-1)
    z_gap = top[:, 2] - bot[:, 2]
    return (xy_dist < xy_threshold) & (torch.abs(z_gap - CUBE_SIZE) < z_threshold)


def _no_contact_with_cube(
    env: "StackCubeTowerEnv",
    cube,
    eps: float = 1e-3,
) -> torch.Tensor:
    """(B,) bool — True per env when neither finger NOR `panda_hand` body has
    any contact force on the given cube (force-norm <= eps on all three).

    Mirrors IsaacLab's `_no_contact_between_cube_and_gripper_or_ee` (which
    reads `force_matrix_w[:, 0, idx, :]` on three contact sensors). The
    `self.scene.get_pairwise_contact_forces(link, actor)` API returns
    `(num_envs, 3)` so the L2-norm collapses to `(num_envs,)`.
    """
    left = env.agent.finger1_link
    right = env.agent.finger2_link
    hand = env.panda_hand_link
    l_force = torch.linalg.norm(env.scene.get_pairwise_contact_forces(left, cube), dim=1)
    r_force = torch.linalg.norm(env.scene.get_pairwise_contact_forces(right, cube), dim=1)
    h_force = torch.linalg.norm(env.scene.get_pairwise_contact_forces(hand, cube), dim=1)
    in_contact = (l_force > eps) | (r_force > eps) | (h_force > eps)
    return ~in_contact


@register_env("StackCube-Tower-v1", max_episode_steps=180)
class StackCubeTowerEnv(BaseEnv):
    """**Task Description:**
    Three identical cubes start lying flat on the table. The policy must build
    a 3-cube tower with ``cube_2`` as the base, ``cube_1`` in the middle, and
    ``cube_0`` on top.

    **Randomizations:**
    - each cube's xy position is sampled inside a non-overlapping per-cube box
    - cube orientation is the IDENTITY quaternion (no yaw randomization —
      matches IsaacLab `EventCfg.reset_cube_*` which only specifies x/y/z
      in `pose_range`).
    - the robot's joint qpos is perturbed by Gaussian noise (``robot_init_qpos_noise``)

    **Success Conditions:**
    - both pairs satisfy ``xy_dist < 0.02`` AND ``|z_gap - CUBE_SIZE| < 0.01``:
      ``cube_0`` on ``cube_1`` AND ``cube_1`` on ``cube_2``.
    - the env does NOT terminate on success (time-out only — full horizon
      always runs); ``info["success"]`` is still reported for the reward.
    """

    SUPPORTED_ROBOTS = ["fr3_franka_hand"]
    agent: Panda

    def __init__(self, *args, robot_uids="fr3_franka_hand", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        # Stash for the obs `last_action` term. The actual dim depends on the
        # controller — we lazy-initialize after super().__init__ inside reset().
        self._last_action: torch.Tensor | None = None
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ------------------------------------------------------------------ #
    # Scene
    # ------------------------------------------------------------------ #
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        # Robot base pose matches IsaacLab `Triton-Franka-StackCube` (which
        # mirrors bidex StackCube): world `(-0.274, 0.49, 0.01)`. Combined
        # with cube reset ranges below, the robot-relative cube geometry
        # matches IsaacLab.
        super()._load_agent(options, sapien.Pose(p=[-0.274, 0.49, 0.01]))

    def _load_scene(self, options: dict):
        self.cube_half_size = common.to_tensor([CUBE_HALF_SIZE] * 3, device=self.device)
        self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        # Distinct colors so the rendered video reads bottom-up: cube_2 (blue
        # base) <- cube_1 (green middle) <- cube_0 (red top). Off-screen
        # initial poses suppress SAPIEN's "no initial pose set" warning.
        self.cube_0 = actors.build_cube(
            self.scene, half_size=CUBE_HALF_SIZE, color=[1.0, 0.0, 0.0, 1.0],
            name="cube_0", initial_pose=sapien.Pose(p=[0.0, 0.0, 0.5]),
        )
        self.cube_1 = actors.build_cube(
            self.scene, half_size=CUBE_HALF_SIZE, color=[0.0, 1.0, 0.0, 1.0],
            name="cube_1", initial_pose=sapien.Pose(p=[0.5, 0.0, 0.5]),
        )
        self.cube_2 = actors.build_cube(
            self.scene, half_size=CUBE_HALF_SIZE, color=[0.0, 0.0, 1.0, 1.0],
            name="cube_2", initial_pose=sapien.Pose(p=[-0.5, 0.0, 0.5]),
        )

        # Pin cube mass to IsaacLab's `CUBE_MASS = 0.055 kg`. `build_cube`
        # defaults to SAPIEN's standard density (~1000 kg/m³) giving ~0.0795 kg
        # for a 0.043 m edge — 45% heavier than IsaacLab. Override per cube.
        for cube in (self.cube_0, self.cube_1, self.cube_2):
            cube.set_mass(CUBE_MASS)

        # §6 reward needs the hand body for the "no contact between
        # cube_0 and gripper/hand" check inside `success_bonus` /
        # `tower_bonus`. With the FR3FrankaHand agent the link is `fr3_hand`
        # (renamed from `panda_hand`). The two finger links are already
        # exposed as `self.agent.finger1_link` / `finger2_link` by the
        # agent's `_after_init`.
        self.panda_hand_link = sapien_utils.get_obj_by_name(
            self.agent.robot.get_links(), "fr3_hand"
        )

        # §6 per-env once-per-episode latches. Created at scene-build time so
        # the buffers persist across `_initialize_episode` calls (which only
        # zero them out for the env indices that are resetting). All initially
        # False — overwritten per-env on each reset.
        self._cube_0_stacked_once = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device,
        )
        self._stack_broke_penalty_fired = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device,
        )
        self._tower_done = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device,
        )

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            # `TableSceneBuilder.initialize` (a) sets Panda default qpos and
            # (b) re-pins the robot base to world `(-0.615, 0, 0)`. We override
            # BOTH below to match IsaacLab.
            self.table_scene.initialize(env_idx)

            # Override 1 — robot base pose: match IsaacLab `Triton-Franka-StackCube`
            # (= bidex StackCube) base placement `(-0.274, 0.49, 0.01)` so the
            # robot-relative cube geometry is identical.
            self.agent.robot.set_pose(sapien.Pose(p=[-0.274, 0.49, 0.01]))

            # Override 2 — init qpos: mirror IsaacLab `FRANKA_INIT_JOINT_POS`
            # (= bidex StackCube init). joint1=-0.785 rotates the arm 45°
            # toward the cubes; joint2/4/6 set a lowered EE arc. Apply small
            # Gaussian noise (robot_init_qpos_noise, default 0.02) on the 7 arm
            # joints only.
            isaaclab_init_qpos = torch.tensor(
                [-0.785, -0.785, 0.0, -2.655, 0.0, 1.87, 0.0, 0.04, 0.04],
                device=self.device, dtype=torch.float32,
            )
            qpos = isaaclab_init_qpos.unsqueeze(0).expand(b, -1).clone()
            qpos[:, :7] = qpos[:, :7] + (
                torch.randn((b, 7), device=self.device) * self.robot_init_qpos_noise
            )
            self.agent.reset(qpos)

            # Per-cube non-overlapping xy boxes mirror IsaacLab's
            # `EventCfg.reset_cube_{0,1,2}` ranges (those events add an
            # offset to `init_state.pos = (0,0,...)`, so the world-frame
            # spawn ranges below are identical to the IsaacLab values).
            #   cube_0: x in [ 0.00,  0.10], y in [ 0.15, 0.25]
            #   cube_1: x in [ 0.00,  0.10], y in [ 0.00, 0.10]
            #   cube_2: x in [-0.15, -0.05], y in [ 0.00, 0.10]
            # Cubes lie flat on the table (z = CUBE_HALF_SIZE).
            def _sample_xy(b, xmin, xmax, ymin, ymax):
                xy = torch.empty((b, 2))
                xy[:, 0] = torch.rand(b) * (xmax - xmin) + xmin
                xy[:, 1] = torch.rand(b) * (ymax - ymin) + ymin
                return xy

            # IsaacLab `EventCfg.reset_cube_*` uses `pose_range` with only
            # x/y/z entries — no roll/pitch/yaw — so cubes are reset with the
            # IDENTITY quaternion. Mirror that here (no yaw randomization).
            identity_q = torch.zeros((b, 4))
            identity_q[:, 0] = 1.0
            for cube, (xmin, xmax, ymin, ymax) in (
                (self.cube_0, (0.00, 0.10, 0.15, 0.25)),
                (self.cube_1, (0.00, 0.10, 0.00, 0.10)),
                (self.cube_2, (-0.15, -0.05, 0.00, 0.10)),
            ):
                xyz = torch.zeros((b, 3))
                xyz[:, :2] = _sample_xy(b, xmin, xmax, ymin, ymax)
                xyz[:, 2] = CUBE_HALF_SIZE
                cube.set_pose(Pose.create_from_pq(p=xyz, q=identity_q))

            # Reset the `last_action` stash to zeros so `_get_obs_extra` has a
            # finite tensor to surface on the very first obs of the episode.
            if self._last_action is None:
                self._last_action = torch.zeros(
                    (self.num_envs, self.agent.action_space.shape[-1]),
                    device=self.device,
                )
            else:
                self._last_action[env_idx] = 0.0

            # §6 once-per-episode latches: clear for the envs that just reset.
            self._cube_0_stacked_once[env_idx] = False
            self._stack_broke_penalty_fired[env_idx] = False
            self._tower_done[env_idx] = False

    # ------------------------------------------------------------------ #
    # Goal + termination (time-out only)
    # ------------------------------------------------------------------ #
    def evaluate(self):
        # Both pairs must satisfy xy<0.02 AND |dz - CUBE_SIZE|<0.01.
        cube_0_on_cube_1 = _cube_0_on_cube_1_predicate(self)
        top = self.cube_1.pose.p
        bot = self.cube_2.pose.p
        xy_dist = torch.linalg.norm(top[:, :2] - bot[:, :2], dim=-1)
        z_gap = top[:, 2] - bot[:, 2]
        cube_1_on_cube_2 = (xy_dist < 0.02) & (torch.abs(z_gap - CUBE_SIZE) < 0.01)
        success = cube_0_on_cube_1 & cube_1_on_cube_2
        return {
            "cube_0_on_cube_1": cube_0_on_cube_1,
            "cube_1_on_cube_2": cube_1_on_cube_2,
            "success": success.bool(),
        }

    def step(self, action):
        """Override to:
          (a) stash the policy action for the ``last_action`` obs term
              (mirrors IsaacLab ``mdp.last_action``);
          (b) force ``terminated`` to all-False (time-out only).

        The ``FR3FrankaHand._controller_configs`` override sets
        ``pos_lower=-0.01, pos_upper=0.01`` on ``pd_ee_target_delta_pos``,
        so a policy action in ``[-1, 1]`` produces an effective per-step
        EE delta of ``±0.01 m`` — matching IsaacLab ``scale=0.01``. No
        pre-scaling needed here.
        """
        if action is not None and not isinstance(action, dict):
            act_tensor = common.to_tensor(action, device=self.device)
            if act_tensor.shape == self._orig_single_action_space.shape:
                act_tensor = common.batch(act_tensor)
            self._last_action = act_tensor.clone()
        obs, reward, terminated, truncated, info = super().step(action)
        # IsaacLab `termination_on_success = False`: never terminate, only
        # truncate at max_episode_steps. TimeLimitWrapper still sets `truncated`.
        terminated = torch.zeros_like(terminated)
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Observation — mirrors IsaacLab `ObservationsCfg.PolicyCfg` term order:
    #   ee_pose(7) + grasping_cube(3) + grasping_target(3) + gripper_pos(2) + last_action(4) = 19
    # ------------------------------------------------------------------ #
    def _get_obs_agent(self):
        """Slot 0: ``ee_pose`` = TCP position(3) + quaternion(4) in robot
        root frame. Mirrors IsaacLab ``mdp.ee_pose_in_robot_root_frame``
        (which reads the ``ee_frame`` FrameTransformer)."""
        return dict(ee_pose=_ee_pose_in_robot_root_frame(self))

    def _get_obs_extra(self, info: dict):
        # Insertion order matters — `_flatten_raw_obs` walks dict items in
        # insertion order and concatenates. Final obs layout (state mode):
        #   agent.ee_pose                   (7,)
        #   extra.grasping_cube_position    (3,)
        #   extra.grasping_target_position  (3,)
        #   extra.gripper_pos               (2,)
        #   extra.last_action               (4,)
        # Total = 19.
        last_action = self._last_action
        if last_action is None:
            last_action = torch.zeros(
                (self.num_envs, self.agent.action_space.shape[-1]),
                device=self.device,
            )
        return dict(
            grasping_cube_position=_grasping_cube_position(self),
            grasping_target_position=_grasping_target_position(self),
            gripper_pos=_gripper_pos(self),
            last_action=last_action,
        )

    # ------------------------------------------------------------------ #
    # Reward (faithful port of IsaacLab Triton-Franka-StackCube RewardsCfg)
    #
    # Composer = SUM of weighted terms. Per-term values are stored in
    # `info["detailed_reward"]` PRE-SCALED so the W&B curves match the
    # IsaacLab side (whose RewardManager logs `weight·term` per step).
    #
    # Grasping-cube mux mirrors §5: state A (cube_0 NOT on cube_1) ⇒
    # grasping_cube = cube_0, target = cube_1.xyz + [0,0,CUBE_SIZE].
    # State B (cube_0 ON cube_1) ⇒ grasping_cube = cube_2,
    # target = cube_0.xyz + [0,0,CUBE_SIZE]. State-B compensation
    # `torch.where(on_stack, M·base + 1.0, base_A)` matches IsaacLab.
    # ------------------------------------------------------------------ #
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # Weights match IsaacLab's RewardsCfg verbatim. dt-scaling is not
        # applied (ManiSkill's `compute_dense_reward` does not multiply by
        # dt the way IsaacLab's RewardManager does). The W&B curves stay
        # comparable in *shape*; a future strict-dt port can rescale here.
        w_reach = 0.02
        w_lift = 0.1
        w_align = 0.32
        w_success_bonus = 200.0
        w_stack_broke_penalty = -200.0
        w_tower_bonus = 2000.0
        w_linear_lift = 0.15

        on_stack = _cube_0_on_cube_1_predicate(self)            # (B,)
        on_stack_f = on_stack.float()

        cube_0 = self.cube_0
        cube_1 = self.cube_1
        cube_2 = self.cube_2
        pos_0 = cube_0.pose.p
        pos_1 = cube_1.pose.p
        pos_2 = cube_2.pose.p
        ee_pos = self.agent.tcp.pose.p

        # --- reach (w=0.02) ---
        # base = 1 - tanh(||ee - grasping_cube|| / std). State B: 30·base + 1.0.
        std_reach = 0.1
        grasping_pos = torch.where(on_stack.unsqueeze(-1), pos_2, pos_0)
        d_ee = torch.linalg.norm(grasping_pos - ee_pos, dim=1)
        reach_base = 1.0 - torch.tanh(d_ee / std_reach)
        reach = torch.where(on_stack, 30.0 * reach_base + 1.0, reach_base)

        # --- lift (w=0.1) ---
        # base = 1.0 if grasping_cube.z > min_h else 0.0. State B: 10·lifted + 1.0.
        min_h_lift = 0.04
        z_grasping = torch.where(on_stack, pos_2[:, 2], pos_0[:, 2])
        lifted = (z_grasping > min_h_lift).float()
        lift = torch.where(on_stack, 10.0 * lifted + 1.0, lifted)

        # --- align (w=0.32) ---
        # state A min_h = 0.04, state B min_h_b = 0.0875 (above the cube_0
        # +cube_1 stack so cube_2's align only fires after it's lifted
        # clear of the existing stack). State B: 50·base + 1.0.
        std_align = 0.08
        min_h_a_align = 0.04
        min_h_b_align = 0.0875
        base_pos = torch.where(on_stack.unsqueeze(-1), pos_0, pos_1)
        target_pos = base_pos.clone()
        target_pos[:, 2] = target_pos[:, 2] + CUBE_SIZE
        d_align = torch.linalg.norm(grasping_pos - target_pos, dim=1)
        effective_min_h = torch.where(
            on_stack,
            torch.full_like(z_grasping, min_h_b_align),
            torch.full_like(z_grasping, min_h_a_align),
        )
        align_lifted = (z_grasping > effective_min_h).float()
        align_base = align_lifted * (1.0 - torch.tanh(d_align / std_align))
        align = torch.where(on_stack, 50.0 * align_base + 1.0, align_base)

        # --- linear_lift_grasping_cube (w=0.15) ---
        # State A: linear ramp z_0 ∈ [0.0215, 0.06], gated on BOTH fingers in
        # contact with cube_0 (force-norm > 1e-3). State B: ramp z_2 ∈
        # [0.0215, 0.1075], gated on both fingers in contact with cube_2.
        # State B compensation: 10·base_b·gate_b + 1.0.
        init_z = 0.0215
        target_z_a = 0.06
        target_z_b = 0.1075
        contact_force_threshold = 1e-3
        base_a = ((pos_0[:, 2] - init_z) / max(target_z_a - init_z, 1e-6)).clamp(0.0, 1.0)
        base_b = ((pos_2[:, 2] - init_z) / max(target_z_b - init_z, 1e-6)).clamp(0.0, 1.0)
        # Per-finger × cube contact forces — same idiom as Panda.is_grasping.
        l_force_0 = torch.linalg.norm(
            self.scene.get_pairwise_contact_forces(self.agent.finger1_link, cube_0), dim=1
        )
        r_force_0 = torch.linalg.norm(
            self.scene.get_pairwise_contact_forces(self.agent.finger2_link, cube_0), dim=1
        )
        l_force_2 = torch.linalg.norm(
            self.scene.get_pairwise_contact_forces(self.agent.finger1_link, cube_2), dim=1
        )
        r_force_2 = torch.linalg.norm(
            self.scene.get_pairwise_contact_forces(self.agent.finger2_link, cube_2), dim=1
        )
        gate_a = ((l_force_0 > contact_force_threshold) & (r_force_0 > contact_force_threshold)).float()
        gate_b = ((l_force_2 > contact_force_threshold) & (r_force_2 > contact_force_threshold)).float()
        linear_lift = torch.where(
            on_stack, 10.0 * base_b * gate_b + 1.0, base_a * gate_a,
        )

        # --- success_bonus (w=200, once per episode) ---
        # Geometric stack of cube_0 on cube_1 + EE >= 4 cm away from cube_0 +
        # no contact between cube_0 and (left finger, right finger, hand).
        xy_thr = 0.02
        z_thr = 0.01
        gripper_away_min = 0.04
        xy_01 = torch.linalg.norm(pos_0[:, :2] - pos_1[:, :2], dim=-1)
        z_01 = pos_0[:, 2] - pos_1[:, 2]
        geometric_01 = (xy_01 < xy_thr) & (torch.abs(z_01 - CUBE_SIZE) < z_thr)
        ee_to_0_dist = torch.linalg.norm(ee_pos - pos_0, dim=1)
        far_enough_0 = ee_to_0_dist > gripper_away_min
        no_contact_0 = _no_contact_with_cube(self, cube_0, eps=contact_force_threshold)
        success_geom_no_contact = geometric_01 & no_contact_0
        cube_0_success_now = success_geom_no_contact & far_enough_0
        fire_success = cube_0_success_now & (~self._cube_0_stacked_once)
        # Latch records "cube_0 has been stacked at any point this episode" —
        # used both by `success_bonus` (don't fire again) and by
        # `stack_broke_penalty` (only fires if the stack existed first).
        self._cube_0_stacked_once = self._cube_0_stacked_once | cube_0_success_now
        success_bonus = fire_success.float()

        # --- stack_broke_penalty (w=-200, once per episode) ---
        # +1.0 the FIRST step after `cube_0_stacked_once` is True AND cube_0
        # is no longer geometrically on cube_1. With weight = -200 this is a
        # one-time penalty for breaking a previously successful stack.
        # Declaration order matters: this MUST be evaluated AFTER the latch
        # write above so it sees the freshest `cube_0_stacked_once`.
        currently_stacked_01 = geometric_01
        broke_fire = (
            self._cube_0_stacked_once
            & (~currently_stacked_01)
            & (~self._stack_broke_penalty_fired)
        )
        self._stack_broke_penalty_fired = self._stack_broke_penalty_fired | broke_fire
        stack_broke_penalty = broke_fire.float()

        # --- tower_bonus (w=2000, once per episode) ---
        # Both pairs geometric + no contact between cube_0 and the gripper/hand
        # AND no contact between cube_2 and the gripper/hand. Once-per-episode.
        xy_20 = torch.linalg.norm(pos_2[:, :2] - pos_0[:, :2], dim=-1)
        z_20 = pos_2[:, 2] - pos_0[:, 2]
        geometric_20 = (xy_20 < xy_thr) & (torch.abs(z_20 - CUBE_SIZE) < z_thr)
        no_contact_2 = _no_contact_with_cube(self, cube_2, eps=contact_force_threshold)
        tower_built = geometric_01 & no_contact_0 & geometric_20 & no_contact_2
        tower_fire = tower_built & (~self._tower_done)
        self._tower_done = self._tower_done | tower_built
        tower_bonus = tower_fire.float()

        # Compose pre-scaled per-term contributions. Sum equals the scalar
        # reward up to floating-point error — asserted by the §6 smoke.
        contrib_reach = w_reach * reach
        contrib_lift = w_lift * lift
        contrib_align = w_align * align
        contrib_success_bonus = w_success_bonus * success_bonus
        contrib_stack_broke_penalty = w_stack_broke_penalty * stack_broke_penalty
        contrib_tower_bonus = w_tower_bonus * tower_bonus
        contrib_linear_lift = w_linear_lift * linear_lift

        info["detailed_reward"] = {
            "reach": contrib_reach,
            "lift": contrib_lift,
            "align": contrib_align,
            "success_bonus": contrib_success_bonus,
            "stack_broke_penalty": contrib_stack_broke_penalty,
            "tower_bonus": contrib_tower_bonus,
            "linear_lift_grasping_cube": contrib_linear_lift,
        }

        reward = (
            contrib_reach
            + contrib_lift
            + contrib_align
            + contrib_success_bonus
            + contrib_stack_broke_penalty
            + contrib_tower_bonus
            + contrib_linear_lift
        )
        # Mute the on_stack_f reference so static analyzers don't flag it:
        # it's kept for downstream debug edit modes that want a single
        # "in state B?" indicator without recomputing the predicate.
        del on_stack_f
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # Per-step max is dominated by `tower_bonus = 2000` on the success
        # frame. Dividing by 2000 keeps the curve smooth in [0, ~1] for the
        # vast majority of frames; the success spike normalizes to ~1.0+.
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 2000.0


# Module-level helpers exported for the mux. Kept module-level (not methods)
# so §6 can call them with the same signatures as IsaacLab's
# `grasping_cube_position_in_robot_root_frame` / `..._target_...`.
def _grasping_cube_position(env: "StackCubeTowerEnv") -> torch.Tensor:
    """Stateless per-step mux: cube_0 (state A) or cube_2 (state B), in robot root frame."""
    cube_0_pos = _cube_pos_in_robot_root_frame(env, "cube_0")
    cube_2_pos = _cube_pos_in_robot_root_frame(env, "cube_2")
    use_cube_2 = _cube_0_on_cube_1_predicate(env).unsqueeze(-1)
    return torch.where(use_cube_2, cube_2_pos, cube_0_pos)


def _grasping_target_position(env: "StackCubeTowerEnv") -> torch.Tensor:
    """Stateless per-step mux: cube_1.xyz+[0,0,CUBE_SIZE] (A) or cube_0.xyz+[0,0,CUBE_SIZE] (B)."""
    target_on_cube_1 = _stack_target_in_robot_root_frame(env, "cube_1")
    target_on_cube_0 = _stack_target_in_robot_root_frame(env, "cube_0")
    use_target_on_cube_0 = _cube_0_on_cube_1_predicate(env).unsqueeze(-1)
    return torch.where(use_target_on_cube_0, target_on_cube_0, target_on_cube_1)


def _ee_pose_in_robot_root_frame(env: "StackCubeTowerEnv") -> torch.Tensor:
    """7-D `[x, y, z, qw, qx, qy, qz]` of the TCP in the robot root frame.

    Mirrors IsaacLab's ``mdp.ee_pose_in_robot_root_frame`` which reads the
    ``ee_frame`` FrameTransformer (panda_hand + [0,0,0.2] offset) and
    transforms world->root via ``subtract_frame_transforms``. We use
    ``self.agent.tcp`` (= ``fr3_hand_tcp`` link, offset 0.209 m from
    fr3_hand per the URDF — 9 mm off from IsaacLab's 0.2, accept).
    """
    root_pose = env.agent.robot.pose            # (B,) Pose, world frame
    tcp_pose_w = env.agent.tcp.pose             # (B,) Pose, world frame
    tcp_pose_root = root_pose.inv() * tcp_pose_w
    # `.p` is (B, 3); `.q` is (B, 4) in `wxyz` order — same as IsaacLab `quat_w`.
    return torch.cat([tcp_pose_root.p, tcp_pose_root.q], dim=-1)


def _gripper_pos(env: "StackCubeTowerEnv") -> torch.Tensor:
    """2-D finger-joint positions, mirrors IsaacLab `mdp.joint_pos(joint_names=fr3_finger.*)`."""
    qpos = env.agent.robot.get_qpos()
    # Per-URDF: finger joints are the LAST two of the 9 DOFs.
    return qpos[:, -2:]
