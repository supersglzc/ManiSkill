"""Dex-Grasp-v1 — single-arm UF850+Allegro grasp+lift, port of IsaacLab `Isaac-Dex-Grasp`.

Mirrors the spec at
`/home/steven/code/agentic/IsaacLab/nautilus/create-task/isaac-dex-grasp-implementation.md`.

A UFactory 850 (6-DoF) + Allegro right hand (16-DoF) sits at env-local
`(-0.274, -0.475, 0.01)` and must grasp + lift a small rigid object
(0.11 kg) sitting on the lab table at env-local `(0.05, -0.35, 0.0)`. The
goal: drive the object to env-local `(0.05, -0.35, 0.30)` (same xy as
spawn, +30 cm in z) within a 10 cm tolerance before the 8.33-s horizon
(166 steps @ 20 Hz) expires.

Action: 22-D EMA cumulative-relative joint position controller (scale=0.03,
alpha=0.2). Observation: 47-D (22 normalized joint_pos + 3 object xyz +
22 last_action). Reward: 5-term ladder
(palm_to_obj → fingertip_to_obj → lift_height → obj_to_target → success_bonus)
with `composer=sum`. The IsaacLab spec also had a `grasp_contact` reward
plus contact-gating on `lift_height` / `obj_to_target`; those have been
removed here so the policy is free to find non-grasp lifting strategies.
"""
from __future__ import annotations
from typing import Any
from pathlib import Path

import numpy as np
import sapien
import torch

from mani_skill.agents.robots.uf850_allegro import UF850AllegroRight
from mani_skill.agents.robots.uf850_allegro.uf850_allegro_right import (
    JOINT_LOWER_LIMIT, JOINT_UPPER_LIMIT, INIT_QPOS,
)
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


# Object mass + target — match IsaacLab spec.
OBJECT_MASS = 0.11
OBJECT_INIT_LOCAL = (0.05, -0.35, 0.0)      # env-local spawn (z=0 → object settles on table)
OBJECT_TARGET_LOCAL = (0.05, -0.35, 0.30)   # +30 cm above spawn
OBJECT_TARGET_TOL = 0.10                     # 10 cm success tolerance

# Object mesh paths.
_OBJ_VISUAL = str(
    Path(__file__).resolve().parents[4] / "nautilus" / "assets" / "dex_grasp_object" / "object.obj"
)
_OBJ_COLLISION = str(
    Path(__file__).resolve().parents[4] / "nautilus" / "assets" / "dex_grasp_object" / "object_collision.obj"
)


def _object_pos(env: "DexGraspEnv") -> torch.Tensor:
    """Object COM xyz in env-local world frame (ManiSkill returns per-env-local poses)."""
    return env.object.pose.p


def _palm_pos(env: "DexGraspEnv") -> torch.Tensor:
    """Allegro palm_link position in world frame."""
    return env.agent.palm_link.pose.p


def _fingertip_pos(env: "DexGraspEnv", link_name: str) -> torch.Tensor:
    """Fingertip link (if5/mf5/pf5/th5) position in world frame."""
    return getattr(env.agent, f"{link_name}_link").pose.p


@register_env("Dex-Grasp-v1", max_episode_steps=166)
class DexGraspEnv(BaseEnv):
    """Single-arm dexterous grasp+lift. See module docstring."""

    SUPPORTED_ROBOTS = ["uf850_allegro_right"]
    agent: UF850AllegroRight

    def __init__(self, *args, robot_uids="uf850_allegro_right", robot_init_qpos_noise: float = 0.0, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self._last_action: torch.Tensor | None = None
        self._success_bonus_fired: torch.Tensor | None = None
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        # Mirror IsaacLab dex_grasp physx caps (high — 16-finger hand + 4096 envs).
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_aggregate_pairs_capacity=2 ** 22,    # 4 M
                total_aggregate_pairs_capacity=2 ** 16,         # 64 K
                max_rigid_contact_count=2 ** 24,                # 16 M (IsaacLab spec)
                max_rigid_patch_count=2 ** 22,                  # 4 M
            ),
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[1.0, 0, 0.6], target=[0.0, -0.2, 0.2])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.0, 0.5, 0.6], [0.0, -0.2, 0.15])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.274, -0.475, 0.01]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()

        # Object — load from the recentered visual + collision OBJ pair.
        builder = self.scene.create_actor_builder()
        builder.add_multiple_convex_collisions_from_file(filename=_OBJ_COLLISION)
        builder.add_visual_from_file(filename=_OBJ_VISUAL)
        builder.initial_pose = sapien.Pose(p=[0.5, 0.0, 0.5])   # off-screen; reset re-places
        self.object = builder.build(name="object")
        self.object.set_mass(OBJECT_MASS)

        self._success_bonus_fired = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Robot base pose + init qpos (IsaacLab bidex right-robot verbatim).
            self.agent.robot.set_pose(sapien.Pose(p=[-0.274, -0.475, 0.01]))
            qpos = torch.tensor(INIT_QPOS, device=self.device, dtype=torch.float32) \
                .unsqueeze(0).expand(b, -1).contiguous()
            # Optional small jitter on all 22 joints.
            if self.robot_init_qpos_noise > 0:
                qpos = qpos + torch.randn_like(qpos) * self.robot_init_qpos_noise
            self.agent.reset(qpos)

            # Object reset — env-local (0.05, -0.35, 0.0) with identity quat.
            obj_xyz = torch.zeros((b, 3), device=self.device)
            obj_xyz[:, 0] = OBJECT_INIT_LOCAL[0]
            obj_xyz[:, 1] = OBJECT_INIT_LOCAL[1]
            obj_xyz[:, 2] = OBJECT_INIT_LOCAL[2]
            obj_q = torch.zeros((b, 4), device=self.device)
            obj_q[:, 0] = 1.0
            self.object.set_pose(Pose.create_from_pq(p=obj_xyz, q=obj_q))

            # Reset latches + last_action.
            self._success_bonus_fired[env_idx] = False
            if self._last_action is None:
                self._last_action = torch.zeros((self.num_envs, 22), device=self.device)
            else:
                self._last_action[env_idx] = 0.0

            # EMA controller partial-reset re-anchor (same pattern as
            # insert_drawer / lift_box).
            if hasattr(self.agent, "controller") and self.agent.controller is not None:
                ctrl = self.agent.controller
                if hasattr(ctrl, "controllers") and "arm_hand" in ctrl.controllers:
                    arm_ctrl = ctrl.controllers["arm_hand"]
                    if hasattr(arm_ctrl, "_needs_reanchor"):
                        arm_ctrl._needs_reanchor[env_idx] = True
                        arm_ctrl.del_action[env_idx] = 0.0
                        arm_ctrl._partial_reset_env_idx = env_idx

    def evaluate(self):
        obj_local = _object_pos(self)
        target = torch.tensor(OBJECT_TARGET_LOCAL, device=self.device, dtype=obj_local.dtype)
        err = torch.norm(obj_local - target.unsqueeze(0), dim=-1)
        success = err < OBJECT_TARGET_TOL
        # Fail termination: dog more than 2 m from the robot base.
        robot_root = self.agent.robot.pose.p
        d_dog_to_root = torch.norm(obj_local - robot_root, dim=-1)
        fail = d_dog_to_root > 2.0
        return {
            "object_err_to_target": err,
            "success": success.bool(),
            "fail": fail.bool(),
        }

    def step(self, action):
        if action is not None and not isinstance(action, dict):
            act_tensor = common.to_tensor(action, device=self.device)
            if act_tensor.shape == self._orig_single_action_space.shape:
                act_tensor = common.batch(act_tensor)
            self._last_action = act_tensor.clone()
        return super().step(action)

    # ---- Observation (47-D: joint_pos_normalized(22) + obj_xyz(3) + last_action(22)) ----
    def _get_obs_agent(self):
        # Normalize joint_pos to [-1, 1] using IsaacLab JOINT_LOWER/UPPER limits.
        qpos = self.agent.robot.get_qpos()
        lower = torch.tensor(JOINT_LOWER_LIMIT, device=self.device, dtype=qpos.dtype)
        upper = torch.tensor(JOINT_UPPER_LIMIT, device=self.device, dtype=qpos.dtype)
        # scale_transform: q in [lower, upper] -> [-1, 1].
        normed = 2.0 * (qpos - lower) / (upper - lower) - 1.0
        return dict(joint_pos_normalized=normed)

    def _get_obs_extra(self, info: dict):
        last_action = self._last_action
        if last_action is None:
            last_action = torch.zeros((self.num_envs, 22), device=self.device)
        return dict(
            object_position=_object_pos(self),
            last_action=last_action,
        )

    # ---- Reward (6 terms, sum composer) — exact IsaacLab port ----
    # Stage 1 palm_to_obj · Stage 2 fingertip_to_obj · Stage 3 grasp_contact
    # (bidex Allegro predicate: thumb force > 1 N AND any of {idx,mid,pinky} > 1 N)
    # · Stage 4 lift_height (gated on grasp_contact) · Stage 5 obj_to_target
    # (gated on grasp_contact AND lifted) · Stage 6 success_bonus (latch).
    def _allegro_grasp_predicate(self, threshold: float = 1.0) -> torch.Tensor:
        """Bool[N] grasp predicate from real contact forces.

        Mirrors IsaacLab `_allegro_grasp_predicate`: thumb in contact AND at
        least one of {index, middle, pinky} in contact, where 'in contact'
        means pairwise contact-force magnitude with the dog > `threshold` N.
        """
        scene = self.scene
        obj = self.object
        agent = self.agent
        forces = []
        for link in (agent.if5_link, agent.mf5_link, agent.pf5_link, agent.th5_link):
            f = torch.norm(scene.get_pairwise_contact_forces(link, obj), dim=-1)
            forces.append(f > threshold)
        in_if, in_mf, in_pf, in_th = forces
        any_non_thumb = in_if | in_mf | in_pf
        return in_th & any_non_thumb

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        w_palm = 0.025
        w_finger = 0.10
        w_grasp = 0.10
        w_lift = 0.50
        w_obj_target = 2.0
        w_success = 100.0

        obj_pos = _object_pos(self)
        palm_pos = _palm_pos(self)

        # Stage 1 — palm_to_obj attractor. Target the palm 0.10 m ABOVE the
        # object so the palm doesn't try to crash into the dog — when palm
        # reaches the offset target, fingertips are close to the object.
        palm_offset = torch.tensor([0.0, 0.0, 0.10], device=self.device, dtype=palm_pos.dtype)
        d_palm = torch.norm(palm_pos - (obj_pos + palm_offset.unsqueeze(0)), dim=-1)
        palm_to_obj = 1.0 - torch.tanh(d_palm / max(0.20, 1e-6))

        # Stage 2 — fingertip-to-obj weighted-mean attractor (thumb 1.5x).
        weights = torch.tensor([1.0, 1.0, 1.0, 1.5], device=self.device)
        w_norm = weights / weights.sum()
        per_finger_reward = []
        for name in ("if5", "mf5", "pf5", "th5"):
            f_pos = _fingertip_pos(self, name)
            d = torch.norm(f_pos - obj_pos, dim=-1)
            per_finger_reward.append(1.0 - torch.tanh(d / max(0.10, 1e-6)))
        per_finger_reward = torch.stack(per_finger_reward, dim=-1)
        fingertip_to_obj = (per_finger_reward * w_norm.unsqueeze(0)).sum(dim=-1)

        # Stage 3 — bidex Allegro grasp predicate (real contact forces, 1 N).
        grasp_pred = self._allegro_grasp_predicate(threshold=1.0)
        grasp_contact = grasp_pred.float()

        # Stage 4 — linear lift ramp on raw dog.z, gated on grasp_contact.
        # IsaacLab params verbatim: init_z=0.0, target_lift=0.30.
        obj_z = obj_pos[:, 2]
        init_z = 0.0
        target_lift = 0.30
        lift_height = ((obj_z - init_z) / max(target_lift, 1e-6)).clamp(0.0, 1.0) * grasp_contact

        # Stage 5 — dense xyz attractor on object -> target, gated on
        # grasp_contact AND obj.z > init_z + 0.05. IsaacLab params verbatim:
        # std=0.15, lift_threshold=0.05, init_z=0.0.
        target = torch.tensor(OBJECT_TARGET_LOCAL, device=self.device, dtype=obj_pos.dtype)
        d_obj = torch.norm(obj_pos - target.unsqueeze(0), dim=-1)
        align_base = 1.0 - torch.tanh(d_obj / max(0.15, 1e-6))
        lifted = (obj_z > (init_z + 0.05)).float()
        obj_to_target = lifted * align_base * grasp_contact

        # Stage 6 — one-shot success bonus (NOT gated on contact, matching
        # IsaacLab; success itself already implies dog reached target).
        success = info["success"] if "success" in info else self.evaluate()["success"]
        fire = success & (~self._success_bonus_fired)
        self._success_bonus_fired = self._success_bonus_fired | success
        success_bonus = fire.float()

        contrib_palm    = w_palm       * palm_to_obj
        contrib_finger  = w_finger     * fingertip_to_obj
        contrib_grasp   = w_grasp      * grasp_contact
        contrib_lift    = w_lift       * lift_height
        contrib_align   = w_obj_target * obj_to_target
        contrib_success = w_success    * success_bonus

        info["detailed_reward"] = {
            "palm_to_obj":      contrib_palm,
            "fingertip_to_obj": contrib_finger,
            "grasp_contact":    contrib_grasp,
            "lift_height":      contrib_lift,
            "obj_to_target":    contrib_align,
            "success_bonus":    contrib_success,
        }
        return (contrib_palm + contrib_finger + contrib_grasp
                + contrib_lift + contrib_align + contrib_success)

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # success_bonus dominates at +100 once per episode.
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 100.0
