"""Lift-Box-v1 — dual-arm cooperative box lift, faithful port of IsaacLab `Triton-Lift-Box`.

Two FR3 + Franka-hand robots stand at world `y = ±0.49` facing each other.
A 0.40 × 0.30 × 0.22 m eurobox (0.5 kg) sits centered on the lab table,
rotated 90° about +Z so its long axis runs along world Y. Each robot grasps
its assigned short y-end face; the goal is to lift the box's COM to
`(0, 0, BOX_INIT_Z + 0.25) = (0, 0, 0.36025)` with `|lin_vel| < 0.10 m/s`.

Episode horizon: 10 s @ 20 Hz = 200 steps.
"""
from __future__ import annotations
from typing import Any, Tuple
from pathlib import Path

import numpy as np
import sapien
import torch

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.panda.fr3_franka_hand import FR3FrankaHand
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


BOX_HALF_X = 0.20    # half-extents of the recentered eurobox (matches IsaacLab)
BOX_HALF_Y = 0.15
BOX_HALF_Z = 0.11025
BOX_INIT_Z = BOX_HALF_Z   # box COM z when sitting on table (z=0)
BOX_MASS = 0.5
LIFT_HEIGHT = 0.25
TARGET_Z = BOX_INIT_Z + LIFT_HEIGHT       # = 0.36025

# Eurobox mesh (recentered so geometric center = origin) bundled in repo.
_EUROBOX_STL = str(
    Path(__file__).resolve().parents[4]
    / "nautilus" / "assets" / "eurobox" / "eurobox.stl"
)

# Per-robot workspace clamps (in robot root frame) — match IsaacLab spec §2.
_WORKSPACE_LOWER_0 = (0.20, -0.65, 0.005)
_WORKSPACE_UPPER_0 = (0.55, -0.20, 0.40)
_WORKSPACE_LOWER_1 = (0.20, +0.20, 0.005)
_WORKSPACE_UPPER_1 = (0.55, +0.65, 0.40)


# ----------------------------------------------------------------------- #
# Helpers (env-local world coords or robot root frame)
# ----------------------------------------------------------------------- #


def _box_pos_local(env: "LiftBoxEnv") -> torch.Tensor:
    """Box xyz in env-local world coords. ManiSkill already returns per-env
    positions in env-local frame (no IsaacLab-style env_origins subtraction)."""
    return env.box.pose.p


def _box_quat(env: "LiftBoxEnv") -> torch.Tensor:
    """Box orientation quaternion (wxyz), shape (B, 4)."""
    return env.box.pose.q


def _box_lin_vel(env: "LiftBoxEnv") -> torch.Tensor:
    """Box linear velocity in world frame, shape (B, 3)."""
    return env.box.linear_velocity


def _ee_pose_root(env: "LiftBoxEnv", robot_idx: int) -> torch.Tensor:
    """7-D EE pose (xyz + wxyz quat) in the i-th robot's root frame."""
    agent = env.agent.agents[robot_idx]
    root_pose = agent.robot.pose
    tcp_pose_w = agent.tcp.pose
    rel = root_pose.inv() * tcp_pose_w
    return torch.cat([rel.p, rel.q], dim=-1)


def _gripper_pos(env: "LiftBoxEnv", robot_idx: int) -> torch.Tensor:
    """Last two qpos entries (= 2 finger joints) for robot i."""
    return env.agent.agents[robot_idx].robot.get_qpos()[:, -2:]


def _grasp_frame_pos_w(env: "LiftBoxEnv", robot_idx: int) -> torch.Tensor:
    """Top-center of the box's short y-end face for robot i, in world frame.

    Box-local offset: `(+BOX_HALF_X, 0, +BOX_HALF_Z)` for robot_0, mirrored
    for robot_1. After the 90° Z-rotation that the box spawns with, the
    box-local +x maps to world +y, so:
        grasp_0_world ≈ box_pos + (0, +0.20, +0.11025)
        grasp_1_world ≈ box_pos + (0, -0.20, +0.11025)
    """
    box_pose_w = env.box.pose
    offset_local = torch.zeros((env.num_envs, 3), device=env.device)
    offset_local[:, 0] = +BOX_HALF_X if robot_idx == 0 else -BOX_HALF_X
    offset_local[:, 2] = +BOX_HALF_Z
    offset_pose = Pose.create_from_pq(p=offset_local, q=None)
    return (box_pose_w * offset_pose).p


def _both_fingers_in_contact(env: "LiftBoxEnv", robot_idx: int, thr: float = 1e-3) -> torch.Tensor:
    """Bool (B,): both fingertips of robot i in contact with the box (force-norm > thr)."""
    agent = env.agent.agents[robot_idx]
    l_force = torch.linalg.norm(
        env.scene.get_pairwise_contact_forces(agent.finger1_link, env.box), dim=1,
    )
    r_force = torch.linalg.norm(
        env.scene.get_pairwise_contact_forces(agent.finger2_link, env.box), dim=1,
    )
    return (l_force > thr) & (r_force > thr)


# ----------------------------------------------------------------------- #
# Env
# ----------------------------------------------------------------------- #


@register_env("Lift-Box-v1", max_episode_steps=200)
class LiftBoxEnv(BaseEnv):
    """Dual-arm cooperative box-lift. See module docstring."""

    SUPPORTED_ROBOTS = [("fr3_franka_hand", "fr3_franka_hand")]
    agent: MultiAgent[Tuple[FR3FrankaHand, FR3FrankaHand]]

    def __init__(
        self,
        *args,
        robot_uids=("fr3_franka_hand", "fr3_franka_hand"),
        robot_init_qpos_noise: float = 0.0,        # IsaacLab uses position_range=(1.0, 1.0), no jitter
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self._last_action: torch.Tensor | None = None
        self._success_bonus_fired: torch.Tensor | None = None
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        # Flatten the MultiAgent Dict action_space into a single Box(8,) so
        # downstream PPO sees a single arm-style action (4 per robot
        # = 8 total). `step()` reverses the flatten -> dict.
        self._agent_uids = list(self.agent.agents_dict.keys())
        per_uid_low = []
        per_uid_high = []
        for uid in self._agent_uids:
            space = self.agent.agents_dict[uid].single_action_space
            per_uid_low.append(space.low.flatten())
            per_uid_high.append(space.high.flatten())
        flat_low = np.concatenate(per_uid_low)
        flat_high = np.concatenate(per_uid_high)
        from gymnasium import spaces
        self.single_action_space = spaces.Box(flat_low, flat_high, dtype=np.float32)
        # Build a batched action_space for num_envs.
        batched_low = np.broadcast_to(flat_low, (self.num_envs, *flat_low.shape))
        batched_high = np.broadcast_to(flat_high, (self.num_envs, *flat_high.shape))
        self.action_space = spaces.Box(batched_low.astype(np.float32), batched_high.astype(np.float32), dtype=np.float32)
        # Note: do NOT override `_orig_single_action_space` here — ManiSkill's
        # `_step_action` (sapien_env.py:1100) still expects the original Dict
        # form for the multi-agent dispatch (`self._orig_single_action_space[k]`).
        # Our `step()` override below converts the policy's flat action into
        # the dict before forwarding, so `_orig_single_action_space` stays as
        # the BaseEnv-set Dict.
        # Per-uid slice indices for un-flattening in step().
        self._uid_action_slices = []
        offset = 0
        for uid in self._agent_uids:
            dim = int(self.agent.agents_dict[uid].single_action_space.shape[-1])
            self._uid_action_slices.append((uid, offset, offset + dim))
            offset += dim
        self._flat_action_dim = offset

    @property
    def _default_sim_config(self):
        # Match the IsaacLab insert_drawer-style GPU buffers — dual-arm + box
        # contact pairs need significant headroom.
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_aggregate_pairs_capacity=2 ** 22,   # 4 M
                total_aggregate_pairs_capacity=2 ** 16,        # 64 K
                max_rigid_contact_count=2 ** 21,               # 2 M
                max_rigid_patch_count=2 ** 20,                 # 1 M
            ),
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[1.4, 0, 0.8], target=[0.0, 0.0, 0.2])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.4, 0.6, 0.8], [0.0, 0.0, 0.2])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        # Both robots at world (-0.274, ±0.49, 0.01) per IsaacLab spec.
        super()._load_agent(
            options,
            [
                sapien.Pose(p=[-0.274, +0.49, 0.01]),
                sapien.Pose(p=[-0.274, -0.49, 0.01]),
            ],
        )

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()

        # Box — load the recentered STL as a rigid actor.
        builder = self.scene.create_actor_builder()
        builder.add_convex_collision_from_file(filename=_EUROBOX_STL)
        builder.add_visual_from_file(filename=_EUROBOX_STL)
        builder.initial_pose = sapien.Pose(p=[0.5, 0.0, 0.5])  # off-screen; reset re-places
        self.box = builder.build(name="box")
        self.box.set_mass(BOX_MASS)

        # Per-env latches.
        self._success_bonus_fired = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Per-robot workspace clamps on the EMA controller.
        for robot_idx, (lower, upper) in enumerate([
            (_WORKSPACE_LOWER_0, _WORKSPACE_UPPER_0),
            (_WORKSPACE_LOWER_1, _WORKSPACE_UPPER_1),
        ]):
            agent_i = self.agent.agents[robot_idx]
            if hasattr(agent_i, "controller") and agent_i.controller is not None:
                ctrl = agent_i.controller
                if hasattr(ctrl, "controllers") and "arm" in ctrl.controllers:
                    arm_ctrl = ctrl.controllers["arm"]
                    if hasattr(arm_ctrl, "_pos_lower"):
                        arm_ctrl._pos_lower = torch.tensor(lower, device=self.device, dtype=torch.float32)
                        arm_ctrl._pos_upper = torch.tensor(upper, device=self.device, dtype=torch.float32)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Override robot base poses + init qpos.
            agent_0 = self.agent.agents[0]
            agent_1 = self.agent.agents[1]
            agent_0.robot.set_pose(sapien.Pose(p=[-0.274, +0.49, 0.01]))
            agent_1.robot.set_pose(sapien.Pose(p=[-0.274, -0.49, 0.01]))

            # IsaacLab init joint poses, mirrored for robot_1.
            init_qpos_0 = torch.tensor(
                [-0.785, -0.785, 0.0, -2.655, 0.0, 1.87, 0.0, 0.04, 0.04],
                device=self.device, dtype=torch.float32,
            )
            init_qpos_1 = torch.tensor(
                [+0.785, -0.785, 0.0, -2.655, 0.0, 1.87, -1.57, 0.04, 0.04],
                device=self.device, dtype=torch.float32,
            )
            qpos_0 = init_qpos_0.unsqueeze(0).expand(b, -1).contiguous()
            qpos_1 = init_qpos_1.unsqueeze(0).expand(b, -1).contiguous()
            agent_0.reset(qpos_0)
            agent_1.reset(qpos_1)

            # Box reset — center xy ±0.03 cm jitter, z pinned to BOX_INIT_Z,
            # 90° about +Z (matches IsaacLab spawn rotation).
            box_xyz = torch.zeros((b, 3), device=self.device)
            box_xyz[:, 0] = (torch.rand(b, device=self.device) - 0.5) * 0.06   # ±0.03 m
            box_xyz[:, 1] = (torch.rand(b, device=self.device) - 0.5) * 0.06
            box_xyz[:, 2] = BOX_INIT_Z
            box_q = torch.zeros((b, 4), device=self.device)
            box_q[:, 0] = 0.7071068
            box_q[:, 3] = 0.7071068
            self.box.set_pose(Pose.create_from_pq(p=box_xyz, q=box_q))

            # Reset per-env latches + last_action.
            self._success_bonus_fired[env_idx] = False
            if self._last_action is None:
                # 8-D: 4 per FR3FrankaHand (3 xyz EE delta + 1 gripper) × 2 robots.
                self._last_action = torch.zeros((self.num_envs, 8), device=self.device)
            else:
                self._last_action[env_idx] = 0.0

            # Re-anchor the EMA controllers ONLY for the resetting envs
            # (see insert_drawer.py for the rationale on why the default
            # ManiSkill controller-reset path is unsafe under partial reset).
            for agent_i in (agent_0, agent_1):
                if hasattr(agent_i, "controller") and agent_i.controller is not None:
                    ctrl = agent_i.controller
                    if hasattr(ctrl, "controllers") and "arm" in ctrl.controllers:
                        arm_ctrl = ctrl.controllers["arm"]
                        if hasattr(arm_ctrl, "_needs_reanchor"):
                            arm_ctrl._needs_reanchor[env_idx] = True
                            arm_ctrl.del_action[env_idx] = 0.0
                            arm_ctrl._partial_reset_env_idx = env_idx

    def evaluate(self):
        box_local = _box_pos_local(self)
        box_xy = box_local[:, :2]
        box_z = box_local[:, 2]
        xy_err = torch.norm(box_xy, dim=-1)
        z_err = torch.abs(box_z - TARGET_Z)
        vel_norm = torch.norm(_box_lin_vel(self), dim=-1)
        success = (xy_err < 0.05) & (z_err < 0.05) & (vel_norm < 0.10)
        return {
            "box_xy_err": xy_err,
            "box_z_err": z_err,
            "box_lin_vel": vel_norm,
            "success": success.bool(),
        }

    def step(self, action):
        # Convert the flat 8-D action into the per-agent dict that
        # `BaseEnv._step_action` expects when `MultiAgent` is the agent.
        if action is not None and not isinstance(action, dict):
            act_tensor = common.to_tensor(action, device=self.device)
            if act_tensor.dim() == 1:                          # single-env (8,)
                act_tensor = act_tensor.unsqueeze(0)
            self._last_action = act_tensor.clone()
            action = {
                uid: act_tensor[:, lo:hi].contiguous()
                for (uid, lo, hi) in self._uid_action_slices
            }
        return super().step(action)

    # ---- Observation (33-D: ee_pose×2 + box_xyz + box_quat + gripper×2 + last_action) ----
    def _get_obs_agent(self):
        # Stash the agent obs slot for ee_pose_0 (the rest go into _get_obs_extra).
        return dict(ee_pose_0=_ee_pose_root(self, 0))

    def _get_obs_extra(self, info: dict):
        last_action = self._last_action
        if last_action is None:
            last_action = torch.zeros(
                (self.num_envs, self.agent.action_space.shape[-1]),
                device=self.device,
            )
        return dict(
            ee_pose_1=_ee_pose_root(self, 1),
            box_position_in_world=_box_pos_local(self),
            box_quat_in_world=_box_quat(self),
            gripper_joint_pos_0=_gripper_pos(self, 0),
            gripper_joint_pos_1=_gripper_pos(self, 1),
            last_action=last_action,
        )

    # ---- Reward (7 terms, sum composer) ----
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # Weights match IsaacLab `RewardsCfg` exactly.
        w_ee0 = w_ee1 = 0.0125
        w_contact0 = w_contact1 = 0.025
        w_lift = 0.1875
        w_align = 0.125
        w_success = 100.0

        ee_0_w = self.agent.agents[0].tcp.pose.p
        ee_1_w = self.agent.agents[1].tcp.pose.p
        grasp_0_w = _grasp_frame_pos_w(self, 0)
        grasp_1_w = _grasp_frame_pos_w(self, 1)
        std_reach = 0.15

        d0 = torch.norm(ee_0_w - grasp_0_w, dim=-1)
        d1 = torch.norm(ee_1_w - grasp_1_w, dim=-1)
        ee_0_to_grasp_0 = 1.0 - torch.tanh(d0 / std_reach)
        ee_1_to_grasp_1 = 1.0 - torch.tanh(d1 / std_reach)

        contact_thr = 1e-3
        gate_0 = _both_fingers_in_contact(self, 0, contact_thr)
        gate_1 = _both_fingers_in_contact(self, 1, contact_thr)
        grasp_contact_0 = gate_0.float()
        grasp_contact_1 = gate_1.float()

        # lift_height — linear ramp gated on BOTH robots in dual contact.
        box_local = _box_pos_local(self)
        box_z = box_local[:, 2]
        progress = ((box_z - BOX_INIT_Z) / max(LIFT_HEIGHT, 1e-6)).clamp(0.0, 1.0)
        dual_contact = (gate_0 & gate_1).float()
        lift_height = progress * dual_contact

        # box_xy_align — tanh attractor on box xy, gated on lift.
        std_align = 0.15
        box_xy = box_local[:, :2]
        d_align = torch.norm(box_xy, dim=-1)
        lift_threshold = 0.05
        lifted = (box_z > (BOX_INIT_Z + lift_threshold)).float()
        box_xy_align = lifted * (1.0 - torch.tanh(d_align / std_align))

        # success_bonus — one-shot fire when success predicate holds.
        success = info["success"] if "success" in info else self.evaluate()["success"]
        fire = success & (~self._success_bonus_fired)
        self._success_bonus_fired = self._success_bonus_fired | success
        success_bonus = fire.float()

        contrib_ee0       = w_ee0       * ee_0_to_grasp_0
        contrib_ee1       = w_ee1       * ee_1_to_grasp_1
        contrib_contact0  = w_contact0  * grasp_contact_0
        contrib_contact1  = w_contact1  * grasp_contact_1
        contrib_lift      = w_lift      * lift_height
        contrib_align     = w_align     * box_xy_align
        contrib_success   = w_success   * success_bonus

        info["detailed_reward"] = {
            "ee_0_to_grasp_0": contrib_ee0,
            "ee_1_to_grasp_1": contrib_ee1,
            "grasp_contact_0": contrib_contact0,
            "grasp_contact_1": contrib_contact1,
            "lift_height":     contrib_lift,
            "box_xy_align":    contrib_align,
            "success_bonus":   contrib_success,
        }
        return (
            contrib_ee0 + contrib_ee1 + contrib_contact0 + contrib_contact1
            + contrib_lift + contrib_align + contrib_success
        )

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # success_bonus dominates at +100 once per episode; normalize by 100.
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 100.0
