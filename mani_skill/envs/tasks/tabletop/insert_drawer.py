"""Insert-Drawer-v1 — faithful port of IsaacLab `Triton-Insert-Drawer`.

Mirrors the spec at
`/home/steven/code/agentic/IsaacLab/nautilus/create-task/triton-insert-drawer-implementation.md`.

Task: Franka FR3 (custom URDF) starts with a drawer ALREADY OPEN (joint pos =
0.30 m). The policy must (1) pick up a small DexCube from the table, (2) lift
it above the drawer rim, (3) place it inside the open drawer, (4) retract the
gripper out of the drawer interior, and (5) push the drawer closed.

Episode ends on either time-out (9 s @ 20 Hz = 180 steps) OR task success
(cube inside latch + drawer joint pos < 0.10 m). Composer = sum, 8 active
reward terms with the IsaacLab ladder
`reach < is_lifted < lift_distance < align < retract < close < cube_inside_latch < success_bonus`.

Robot: FR3 + Franka-hand (`fr3_franka_hand` agent) with action
`pd_ee_ema_delta_pos` (scale=0.01, alpha=0.5, RPY locked, workspace clamp
`[0.34, -0.8, 0.005]` / `[0.50, -0.05, 0.30]` in robot root frame).

Observation (19-dim, robot root frame):

    ee_pose                  (7,)  — TCP pos + quat
    cube_position            (3,)  — zero-masked once cube_inside latch fires
    drawer_body_position     (3,)  — sliding-tray body in robot root frame
    gripper_pos              (2,)  — 2 finger joint positions
    last_action              (4,)
"""
from __future__ import annotations
from typing import Any
from pathlib import Path

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Panda                       # type alias for the agent attribute
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


# Cube edge length / mass — matches IsaacLab `CUBE_SIZE / CUBE_MASS`.
CUBE_SIZE = 0.043
CUBE_HALF_SIZE = CUBE_SIZE / 2.0
CUBE_MASS = 0.055

# Drawer URDF path. Bundled inside the ManiSkill repo so the asset is portable.
_DRAWER_URDF = str(
    Path(__file__).resolve().parents[4]
    / "nautilus" / "assets" / "drawer_no_handle" / "drawer_no_handle.urdf"
)

# Workspace clamp matches IsaacLab `pos_lower_limit / pos_upper_limit`
# in robot root frame (the action's reference frame).
_WORKSPACE_LOWER = (0.34, -0.8, 0.005)
_WORKSPACE_UPPER = (0.50, -0.05, 0.30)

# Per-episode cap on the cumulative `close_drawer` weighted contribution.
# Without this, `close_drawer` can accumulate `weight=100` per step over many
# frames into the multi-thousand range — large enough to make the value
# function unstable. Cap matches the magnitude of the `success_bonus`
# (also +2000) so close-drawer's reward shaping is bounded by the same
# landmark.
_CLOSE_DRAWER_EPISODE_CAP = 2000.0


# ----------------------------------------------------------------------- #
# Helpers (in robot root frame) — exported for §6 reward
# ----------------------------------------------------------------------- #


def _cube_pos_root(env: "InsertDrawerEnv") -> torch.Tensor:
    """cube xyz in robot root frame (B, 3)."""
    return (env.agent.robot.pose.inv() * env.cube_0.pose).p


def _drawer_body_pos_root(env: "InsertDrawerEnv") -> torch.Tensor:
    """Sliding-tray body (`drawer` link) xyz in robot root frame (B, 3)."""
    # find_links_by_name on a ManiSkill Articulation returns the Link object;
    # `.pose` returns a (B,) Pose in world.
    body_pose_w = env.drawer_body_link.pose
    return (env.agent.robot.pose.inv() * body_pose_w).p


def _ee_pose_root(env: "InsertDrawerEnv") -> torch.Tensor:
    """7-D `[x, y, z, qw, qx, qy, qz]` of TCP in robot root frame."""
    tcp_pose_root = env.agent.robot.pose.inv() * env.agent.tcp.pose
    return torch.cat([tcp_pose_root.p, tcp_pose_root.q], dim=-1)


def _gripper_pos(env: "InsertDrawerEnv") -> torch.Tensor:
    """Last two qpos entries — fr3_finger_joint{1,2}."""
    return env.agent.robot.get_qpos()[:, -2:]


def _drawer_joint_pos(env: "InsertDrawerEnv") -> torch.Tensor:
    """Prismatic `base_drawer_joint` position (B,)."""
    return env.drawer.get_qpos()[:, 0]


def _cube_inside_drawer_geometric(
    env: "InsertDrawerEnv",
    xy_threshold: float = 0.20,
    z_rel_floor: float = -0.02,
    z_rel_ceiling: float = 0.07,
) -> torch.Tensor:
    """True per env when the cube is geometrically inside the drawer interior."""
    cube_pos_w = env.cube_0.pose.p              # (B, 3)
    drawer_pos_w = env.drawer_body_link.pose.p  # (B, 3)
    rel = cube_pos_w - drawer_pos_w
    xy_in = torch.norm(rel[:, :2], dim=-1) < xy_threshold
    z_in = (rel[:, 2] > z_rel_floor) & (rel[:, 2] < z_rel_ceiling)
    return xy_in & z_in


def _drop_frame_pos_w(env: "InsertDrawerEnv") -> torch.Tensor:
    """`drawer_body + drawer-local offset (0, 0, 0.25)` in world frame.

    Mirrors IsaacLab `drawer_drop_frame` (FrameTransformer offset z=+0.25).
    """
    drawer_body_pose_w = env.drawer_body_link.pose
    offset_local = torch.zeros((env.num_envs, 3), device=env.device)
    offset_local[:, 2] = 0.25
    # Apply drawer body rotation to the local offset before adding to world pos.
    # Pose composition: drawer_body_pose * Pose(p=offset_local) gives the
    # offset-frame pose in world.
    offset_pose = Pose.create_from_pq(p=offset_local, q=None)  # identity quat
    return (drawer_body_pose_w * offset_pose).p


def _front_face_pos_w(env: "InsertDrawerEnv") -> torch.Tensor:
    """`drawer_body + drawer-local offset (0.17, 0, 0.15)` in world frame.

    Mirrors IsaacLab `drawer_front_face_frame` (front +X-local face center).
    """
    drawer_body_pose_w = env.drawer_body_link.pose
    offset_local = torch.zeros((env.num_envs, 3), device=env.device)
    offset_local[:, 0] = 0.17
    offset_local[:, 2] = 0.15
    offset_pose = Pose.create_from_pq(p=offset_local, q=None)
    return (drawer_body_pose_w * offset_pose).p


# ----------------------------------------------------------------------- #
# Env
# ----------------------------------------------------------------------- #


@register_env("Insert-Drawer-v1", max_episode_steps=180)
class InsertDrawerEnv(BaseEnv):
    """Pick-place-into-open-drawer + close-drawer task. See module docstring."""

    SUPPORTED_ROBOTS = ["fr3_franka_hand"]
    agent: Panda

    @property
    def _default_sim_config(self):
        # Raise GPU contact-pair buffer capacities. ManiSkill defaults are
        # too small for `num_envs=4096` with a 7-link drawer articulation +
        # cube + 9-DoF robot: the SAPIEN GPU contact query overflows and
        # crashes the contact query with CUDA illegal-memory-access.
        # `max_rigid_contact_count` was raised after SAPIEN reported
        # "Contact buffer overflow detected, please increase its size to
        # at least 737280 in the scene desc".
        # The two aggregate-pair caps mirror IsaacLab's
        # `insert_drawer_env_cfg.py` overrides (4M / 64K).
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_aggregate_pairs_capacity=2 ** 22,   # 4 M
                total_aggregate_pairs_capacity=2 ** 16,        # 64 K
                max_rigid_contact_count=2 ** 21,               # 2 M (default 2**19 = 512K)
                max_rigid_patch_count=2 ** 20,                 # 1 M
            ),
        )

    def __init__(self, *args, robot_uids="fr3_franka_hand", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self._last_action: torch.Tensor | None = None
        # Per-env latch state for `cube_inside_bonus_once_per_episode` and
        # phase-2 zero-masking; created on first scene build.
        self._cube_inside_latch: torch.Tensor | None = None
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ---- Scene cameras (cosmetic) ----
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    # ---- Scene build ----
    def _load_agent(self, options: dict):
        # Robot base pose mirrors IsaacLab: world (-0.274, 0.49, 0.01).
        super()._load_agent(options, sapien.Pose(p=[-0.274, 0.49, 0.01]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()

        # Cube — single DexCube-style red cube, identical to StackCube-Tower.
        self.cube_0 = actors.build_cube(
            self.scene,
            half_size=CUBE_HALF_SIZE,
            color=[1.0, 0.0, 0.0, 1.0],
            name="cube_0",
            initial_pose=sapien.Pose(p=[0.0, 0.0, 0.5]),
        )
        self.cube_0.set_mass(CUBE_MASS)

        # Drawer — load the no-handle URDF as a SAPIEN articulation.
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True       # drawer's `base_link` is the cabinet shell — pinned
        loader.name = "drawer"
        articulations = loader.parse(_DRAWER_URDF)["articulation_builders"]
        builder = articulations[0]
        builder.initial_pose = sapien.Pose(p=[0.5, 0.0, 0.5])  # off-screen; reset re-places it
        self.drawer = builder.build(name="drawer")

        # Override the prismatic drawer joint's dynamics to match IsaacLab
        # `actuators={"drawer_slide": ImplicitActuatorCfg(effort_limit=87.0,
        #   stiffness=0.0, damping=1.0, friction=2.0)}` in
        # `insert_drawer/config/franka/joint_pos_env_cfg.py`. The shipped URDF
        # encodes effort=0.1 (too weak) and friction=1.0 (SAPIEN parses to
        # 0.05 — 40× less than IsaacLab's 2.0). Without the override the
        # drawer slides freely after the cube is inserted and the policy can
        # never close it fully.
        drawer_joint = self.drawer.get_active_joints()[0]
        drawer_joint.set_friction(2.0)
        drawer_joint.set_drive_properties(stiffness=0.0, damping=1.0, force_limit=87.0)

        # Cache the sliding-tray link (`drawer` link inside the URDF) for §6 / §5.
        self.drawer_body_link = sapien_utils.get_obj_by_name(self.drawer.get_links(), "drawer")
        # Cache the hand link for `is_grasping`-style checks if needed downstream.
        self.fr3_hand_link = sapien_utils.get_obj_by_name(
            self.agent.robot.get_links(), "fr3_hand"
        )

        # Per-env latch.
        self._cube_inside_latch = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Per-episode running total of the `close_drawer` weighted contribution
        # — used to cap the cumulative close-drawer reward at
        # `_CLOSE_DRAWER_EPISODE_CAP` (see `compute_dense_reward`).
        self._close_drawer_cumulative = torch.zeros(self.num_envs, device=self.device)

        # Apply the workspace clamp to the EMA controller (matches IsaacLab
        # `pos_lower_limit / pos_upper_limit`). The controller was created
        # during agent init; we patch its tensors here once the device is known.
        if hasattr(self.agent, "controller") and self.agent.controller is not None:
            ctrl = self.agent.controller
            if hasattr(ctrl, "controllers") and "arm" in ctrl.controllers:
                arm_ctrl = ctrl.controllers["arm"]
                if hasattr(arm_ctrl, "_pos_lower"):
                    arm_ctrl._pos_lower = torch.tensor(
                        _WORKSPACE_LOWER, device=self.device, dtype=torch.float32,
                    )
                    arm_ctrl._pos_upper = torch.tensor(
                        _WORKSPACE_UPPER, device=self.device, dtype=torch.float32,
                    )

    # ---- Reset ----
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Robot base pose + init qpos — mirror IsaacLab FRANKA_INIT_JOINT_POS.
            # Note: insert_drawer uses joint7 = 1.57 (90° from stack_cube's 0.0).
            self.agent.robot.set_pose(sapien.Pose(p=[-0.274, 0.49, 0.01]))
            isaaclab_init_qpos = torch.tensor(
                [-0.785, -0.785, 0.0, -2.655, 0.0, 1.87, 1.57, 0.04, 0.04],
                device=self.device, dtype=torch.float32,
            )
            qpos = isaaclab_init_qpos.unsqueeze(0).expand(b, -1).clone()
            qpos[:, :7] = qpos[:, :7] + (
                torch.randn((b, 7), device=self.device) * self.robot_init_qpos_noise
            )
            self.agent.reset(qpos)

            # Cube spawn — IsaacLab range (world frame, identity quat).
            cube_xyz = torch.zeros((b, 3))
            cube_xyz[:, 0] = torch.rand(b) * 0.1 + 0.1     # x ∈ [0.1, 0.2]
            cube_xyz[:, 1] = torch.rand(b) * 0.1 + 0.3     # y ∈ [0.3, 0.4]
            cube_xyz[:, 2] = CUBE_HALF_SIZE
            identity_q = torch.zeros((b, 4))
            identity_q[:, 0] = 1.0
            self.cube_0.set_pose(Pose.create_from_pq(p=cube_xyz, q=identity_q))

            # Drawer pose — `pos_range x∈[0.1, 0.2], y=-0.3` plus the 90° about
            # +Z rotation baked into IsaacLab init_state (q = (0.7071, 0, 0, 0.7071)).
            drawer_xyz = torch.zeros((b, 3))
            drawer_xyz[:, 0] = torch.rand(b) * 0.1 + 0.1   # x ∈ [0.1, 0.2]
            drawer_xyz[:, 1] = -0.3
            drawer_xyz[:, 2] = 0.10                        # z=0.10 from IsaacLab
            drawer_q = torch.zeros((b, 4))
            drawer_q[:, 0] = 0.7071068
            drawer_q[:, 3] = 0.7071068
            self.drawer.set_pose(Pose.create_from_pq(p=drawer_xyz, q=drawer_q))

            # Drawer joint — pin OPEN at 0.30 m.
            drawer_qpos = torch.full((b, 1), 0.30, device=self.device)
            self.drawer.set_qpos(drawer_qpos)

            # Reset all per-episode latches for the resetting envs.
            self._cube_inside_latch[env_idx] = False
            if hasattr(self, "_prev_latch"):
                self._prev_latch[env_idx] = False
            if hasattr(self, "_success_bonus_fired"):
                self._success_bonus_fired[env_idx] = False
            # Reset the episodic `close_drawer` cap accumulator.
            self._close_drawer_cumulative[env_idx] = 0.0
            if self._last_action is None:
                self._last_action = torch.zeros(
                    (self.num_envs, self.agent.action_space.shape[-1]),
                    device=self.device,
                )
            else:
                self._last_action[env_idx] = 0.0

            # Mark the EMA controller's `init_ee_pos` for **only the resetting
            # envs** to be re-anchored on the next set_action. ManiSkill's
            # default reset path calls `controller.reset()` AFTER it has
            # overwritten `scene._reset_mask` with all-True
            # (`sapien_env.py:953`), so a generic controller can't distinguish
            # partial from full reset and ends up re-anchoring every env --
            # which wipes the in-flight EMA trajectory state for envs that
            # didn't actually reset. Bypass that here while the env_idx mask
            # is still correct.
            arm_ctrl = None
            if hasattr(self.agent, "controller") and self.agent.controller is not None:
                ctrl = self.agent.controller
                if hasattr(ctrl, "controllers") and "arm" in ctrl.controllers:
                    arm_ctrl = ctrl.controllers["arm"]
            if arm_ctrl is not None and hasattr(arm_ctrl, "_needs_reanchor"):
                arm_ctrl._needs_reanchor[env_idx] = True
                arm_ctrl.del_action[env_idx] = 0.0
                # Stash env_idx so the controller's reset() can no-op the
                # subsequent ManiSkill-driven full re-anchor.
                arm_ctrl._partial_reset_env_idx = env_idx

    # ---- Termination ----
    def evaluate(self):
        """Success when cube is inside-latched AND drawer is closed (joint < 0.10)."""
        # Update the latch this frame (mirrors `cube_inside_bonus_once_per_episode`).
        inside = _cube_inside_drawer_geometric(self)
        ee_w = self.agent.tcp.pose.p
        ee_far = torch.norm(ee_w - self.cube_0.pose.p, dim=-1) > 0.10
        now_qualifying = inside & ee_far
        self._cube_inside_latch = self._cube_inside_latch | now_qualifying

        drawer_jp = _drawer_joint_pos(self)
        drawer_closed = drawer_jp < 0.10
        success = self._cube_inside_latch & drawer_closed
        return {
            "cube_inside_latch": self._cube_inside_latch,
            "drawer_closed": drawer_closed,
            "success": success.bool(),
        }

    def step(self, action):
        """Stash last_action for the obs term. Success termination is honored
        natively by BaseEnv.step via `info["success"]` (IsaacLab also has
        `success` as a terminal DoneTerm with `time_out=False`)."""
        if action is not None and not isinstance(action, dict):
            act_tensor = common.to_tensor(action, device=self.device)
            if act_tensor.shape == self._orig_single_action_space.shape:
                act_tensor = common.batch(act_tensor)
            self._last_action = act_tensor.clone()
        return super().step(action)

    # ---- Observation ----
    def _get_obs_agent(self):
        return dict(ee_pose=_ee_pose_root(self))

    def _get_obs_extra(self, info: dict):
        last_action = self._last_action
        if last_action is None:
            last_action = torch.zeros(
                (self.num_envs, self.agent.action_space.shape[-1]),
                device=self.device,
            )
        # Zero-mask the cube position once the latch fires (matches IsaacLab).
        cube_pos = _cube_pos_root(self)
        mask = (1.0 - self._cube_inside_latch.float()).unsqueeze(-1)
        cube_pos_masked = cube_pos * mask
        return dict(
            cube_position=cube_pos_masked,
            drawer_body_position=_drawer_body_pos_root(self),
            gripper_pos=_gripper_pos(self),
            last_action=last_action,
        )

    # ---- Reward (8 terms, sum composer; per IsaacLab spec) ----
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # Weights mirror IsaacLab RewardsCfg verbatim.
        w_reach = 0.02
        w_is_lifted = 0.2
        w_lift_distance = 0.3
        w_align = 2.0
        w_retract = 2.0
        w_cube_inside_latch = 300.0
        w_close_drawer = 100.0
        w_success_bonus = 2000.0

        # Phase-2 gate: dense terms zero out after the latch fires.
        latch = self._cube_inside_latch.float()       # (B,)
        not_latched = 1.0 - latch

        cube_pos_w = self.cube_0.pose.p              # (B, 3)
        ee_pos_w = self.agent.tcp.pose.p
        drop_pos_w = _drop_frame_pos_w(self)
        front_pos_w = _front_face_pos_w(self)

        # reach_cube — 1 - tanh(||cube - ee|| / 0.1), gated by not_latched.
        std_reach = 0.1
        d_ee = torch.norm(cube_pos_w - ee_pos_w, dim=1)
        reach = (1.0 - torch.tanh(d_ee / std_reach)) * not_latched

        # is_lifted — cube.z > 0.04, gated by not_latched.
        cube_z = cube_pos_w[:, 2]
        is_lifted = (cube_z > 0.04).float() * not_latched

        # lift_distance — ramp on cube.z ∈ [0.0215, 0.25], gated on BOTH
        # finger contact-sensor forces > threshold. Mirrors IsaacLab
        # `lift_distance(contact_force_threshold=1e-3)` exactly. The early
        # geometric proxy `ee_to_cube < 0.05 AND cube_off_table` was a
        # chicken-and-egg: cube can't lift without grasp, grasp gets no
        # reward without lift -> policy never learns. The contact-sensor
        # gate fires the moment the policy SQUEEZES the cube (even before
        # any lift), driving the grasping behaviour directly.
        init_z, target_z = 0.0215, 0.25
        contact_thr = 1e-3
        base_lift = ((cube_z - init_z) / max(target_z - init_z, 1e-6)).clamp(0.0, 1.0)
        l_force = torch.linalg.norm(
            self.scene.get_pairwise_contact_forces(self.agent.finger1_link, self.cube_0), dim=1,
        )
        r_force = torch.linalg.norm(
            self.scene.get_pairwise_contact_forces(self.agent.finger2_link, self.cube_0), dim=1,
        )
        grasp_gate = ((l_force > contact_thr) & (r_force > contact_thr)).float()
        lift_distance = base_lift * grasp_gate * not_latched

        # align — cube → drop_frame attractor (xyz, std=0.20), gated on cube.z >
        # 0.25 AND not_latched.
        std_align = 0.20
        min_h_b = 0.25
        d_align = torch.norm(cube_pos_w - drop_pos_w, dim=-1)
        base_align = 1.0 - torch.tanh(d_align / std_align)
        high_enough = (cube_z > min_h_b).float()
        align = high_enough * base_align * not_latched

        # ee_retract_to_front_face — y-axis-only attractor on EE to drawer
        # front face. Gated by latch (fires only after cube is inserted).
        std_retract = 0.05
        ee_y = ee_pos_w[:, 1]
        front_y = front_pos_w[:, 1]
        dy = torch.abs(ee_y - front_y)
        retract = (1.0 - torch.tanh(dy / std_retract)) * latch

        # cube_inside_bonus_once_per_episode — +1.0 the FIRST frame the
        # geometric+ee-far criterion holds (the latch update happens in
        # evaluate() this step). To make this once-per-episode here, compare
        # the LATCH STATE against last step's latch — but evaluate() has
        # already updated it. Use a separate "just-fired" tensor.
        inside = _cube_inside_drawer_geometric(self)
        ee_far = torch.norm(ee_pos_w - cube_pos_w, dim=-1) > 0.10
        now_qualifying = inside & ee_far
        # The latch was already True for this env if it qualified ANY previous
        # step. To detect FIRST-fire this step: now_qualifying & ~(latch state
        # BEFORE this step's evaluate update). But we updated latch in evaluate.
        # Track this via a "previous-frame" buffer.
        if not hasattr(self, "_prev_latch"):
            self._prev_latch = torch.zeros_like(self._cube_inside_latch)
        fire_inside = now_qualifying & (~self._prev_latch)
        self._prev_latch = self._cube_inside_latch.clone()
        cube_inside_bonus = fire_inside.float()

        # close_drawer — latch * (ee.y > front_y) * closeness^alpha.
        max_open, alpha = 0.30, 1.0
        drawer_jp = _drawer_joint_pos(self)
        closeness = ((max_open - drawer_jp) / max(max_open, 1e-6)).clamp(0.0, 1.0).pow(alpha)
        gate_ee_outside = (ee_y > front_y).float()
        close_drawer = latch * gate_ee_outside * closeness

        # success_bonus — +1.0 the FIRST frame `latch AND drawer_closed` holds.
        # ManiSkill's env_wrapper uses `ignore_terminations=True` (training-time
        # convenience for clean GAE boundaries), so the episode does NOT
        # terminate on success. We latch the bonus so it fires exactly once
        # per episode — matching IsaacLab's one-shot terminal +2000 behavior.
        drawer_closed_bool = drawer_jp < 0.10
        now_success = self._cube_inside_latch & drawer_closed_bool
        if not hasattr(self, "_success_bonus_fired"):
            self._success_bonus_fired = torch.zeros_like(self._cube_inside_latch)
        fire_success = now_success & (~self._success_bonus_fired)
        self._success_bonus_fired = self._success_bonus_fired | now_success
        success_bonus = fire_success.float()

        # Compose.
        contrib_reach           = w_reach           * reach
        contrib_is_lifted       = w_is_lifted       * is_lifted
        contrib_lift_distance   = w_lift_distance   * lift_distance
        contrib_align           = w_align           * align
        contrib_retract         = w_retract         * retract
        contrib_cube_inside     = w_cube_inside_latch * cube_inside_bonus
        contrib_close_drawer    = w_close_drawer    * close_drawer
        # Cap per-episode cumulative `close_drawer` contribution at
        # `_CLOSE_DRAWER_EPISODE_CAP` (= 2000). Once the running total
        # reaches the cap, subsequent close_drawer reward is zero for the
        # rest of the episode. Reset happens in `_initialize_episode`.
        remaining_cap = (_CLOSE_DRAWER_EPISODE_CAP - self._close_drawer_cumulative).clamp(min=0.0)
        contrib_close_drawer = torch.minimum(contrib_close_drawer, remaining_cap)
        self._close_drawer_cumulative = self._close_drawer_cumulative + contrib_close_drawer
        contrib_success_bonus   = w_success_bonus   * success_bonus

        info["detailed_reward"] = {
            "reach_cube":               contrib_reach,
            "is_lifted":                contrib_is_lifted,
            "lift_distance":            contrib_lift_distance,
            "align":                    contrib_align,
            "ee_retract_to_front_face": contrib_retract,
            "cube_inside_bonus_latch":  contrib_cube_inside,
            "close_drawer":             contrib_close_drawer,
            "success_bonus":            contrib_success_bonus,
        }
        return (
            contrib_reach + contrib_is_lifted + contrib_lift_distance + contrib_align
            + contrib_retract + contrib_cube_inside + contrib_close_drawer + contrib_success_bonus
        )

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # Per-step max dominated by success_bonus=2000 on the success frame.
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 2000.0
