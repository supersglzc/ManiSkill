"""EMA cumulative-delta task-space EE-position controller for ManiSkill.

Direct port of IsaacLab's
``isaaclab_tasks.manager_based.manipulation.stack_cube.mdp.actions.EMACumulativeDeltaPositionAction``
into ManiSkill's controller framework. Used by ``StackCube-Tower-v1`` so the
control loop semantics match IsaacLab's ``Triton-Franka-StackCube`` task.

Per-step processing inside ``set_action`` (3-D policy action ``a``, scale ``s``,
alpha ``α``, init EE pose captured lazily on first step after reset):

    a       = clamp(a, -1, 1)
    delta_t = delta_{t-1} + s · a                                  # cumulative
    abs_pos = init_ee_pos + delta_t
    target  = α · abs_pos + (1 - α) · prev_applied_pos
    target  = clamp(target, pos_lower_limit, pos_upper_limit)      # optional
    cmd     = (target, init_ee_quat)                                # quat locked
    IK(cmd)
    prev_applied_pos = target

State across steps (per env): ``del_action``, ``_prev_applied_pos``,
``init_ee_pos``, ``init_ee_quat``, ``_needs_reanchor``.

On ``reset(env_ids)``: ``del_action[env_ids] := 0``,
``_needs_reanchor[env_ids] := True`` — the NEXT ``set_action`` re-captures
``init_ee_{pos,quat}`` from the (now-fresh) EE pose.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np
import torch

from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array

from .pd_ee_pose import PDEEPosController, PDEEPosControllerConfig


class PDEEEMACumulativeDeltaPosController(PDEEPosController):
    """Position-only EE controller with per-episode EMA-smoothed cumulative delta.

    Action layout: ``[dx, dy, dz]`` in ``[-1, 1]^3`` (the gripper joint stays on
    its own controller alongside).
    """

    config: "PDEEEMACumulativeDeltaPosControllerConfig"

    def _initialize_joints(self):
        super()._initialize_joints()
        n_envs = self.scene.num_envs
        device = self.device
        # Per-env state — same shapes as IsaacLab's EMACumulativeDeltaPositionAction.
        self.del_action = torch.zeros((n_envs, 3), device=device)
        self.init_ee_pos = torch.zeros((n_envs, 3), device=device)
        self.init_ee_quat = torch.zeros((n_envs, 4), device=device)
        self.init_ee_quat[:, 0] = 1.0
        self._prev_applied_pos = torch.zeros((n_envs, 3), device=device)
        self._needs_reanchor = torch.ones(n_envs, dtype=torch.bool, device=device)
        # Optional per-axis workspace clamp (in robot root frame).
        if self.config.pos_lower_limit is not None and self.config.pos_upper_limit is not None:
            self._pos_lower = torch.tensor(self.config.pos_lower_limit, device=device, dtype=torch.float32)
            self._pos_upper = torch.tensor(self.config.pos_upper_limit, device=device, dtype=torch.float32)
        else:
            self._pos_lower = None
            self._pos_upper = None
        # Per-axis scale (broadcast from cfg.scale which may be float or 3-tuple).
        self._scale = torch.tensor(
            self.config.scale if isinstance(self.config.scale, (list, tuple))
            else [self.config.scale] * 3,
            device=device, dtype=torch.float32,
        )
        if self.config.alpha < 0.0 or self.config.alpha > 1.0:
            raise ValueError(f"alpha must be in [0, 1]; got {self.config.alpha}")
        self._alpha = float(self.config.alpha)

    def _initialize_action_space(self):
        # Override parent: action range fixed at [-1, 1]^3 regardless of
        # ``pos_lower / pos_upper`` (those are workspace clamps, not action limits).
        from gymnasium import spaces
        low = np.full(3, -1.0, dtype=np.float32)
        high = np.full(3, 1.0, dtype=np.float32)
        self.single_action_space = spaces.Box(low, high, dtype=np.float32)

    def reset(self):
        super().reset()
        if not hasattr(self, "_needs_reanchor"):
            return
        # ManiSkill's reset path resets `scene._reset_mask` to all-True
        # BEFORE calling `controller.reset()` (see `sapien_env.py:953`),
        # so the mask alone can't distinguish partial from full reset.
        # The env that owns this controller is expected to stash the true
        # env_idx in `self._partial_reset_env_idx` from inside its
        # `_initialize_episode` (where the mask is still correct).
        # If that hint is present, re-anchor ONLY those envs and clear the
        # hint; otherwise re-anchor all (safe full-reset behaviour).
        partial = getattr(self, "_partial_reset_env_idx", None)
        if partial is not None:
            self._needs_reanchor[partial] = True
            self.del_action[partial] = 0.0
            self._partial_reset_env_idx = None
            return
        self._needs_reanchor[:] = True
        self.del_action[:] = 0.0

    def _preprocess_action(self, action: Array) -> torch.Tensor:
        # Skip parent's clip+scale; we apply our own clip + IsaacLab-style scale.
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, device=self.device, dtype=torch.float32)
        else:
            action = action.to(self.device).float()
        return action

    def set_action(self, action: Array):
        # 3-D policy action (xyz delta). Apply the IsaacLab EMA + cumulative
        # algorithm, then call kinematics.compute_ik() to drive the arm.
        action = self._preprocess_action(action)

        # Re-anchor any env that just reset — pull init_ee_{pos,quat} from the
        # current EE pose (which is now post-reset and stable).
        if self._needs_reanchor.any():
            cur_pose = self.ee_pose_at_base
            cur_pos = cur_pose.p          # (N, 3)
            cur_quat = cur_pose.q         # (N, 4) wxyz
            mask = self._needs_reanchor
            self.init_ee_pos[mask] = cur_pos[mask]
            self.init_ee_quat[mask] = cur_quat[mask]
            self._prev_applied_pos[mask] = cur_pos[mask]
            self._needs_reanchor[:] = False

        # Clip to [-1, 1] (matches IsaacLab `torch.clamp(actions, -1.0, 1.0)`).
        action = torch.clamp(action, -1.0, 1.0)

        # Scaled per-step delta (3-D), accumulated.
        scaled = action * self._scale
        self.del_action = self.del_action + scaled

        # Clamp the cumulative delta to the workspace bounds, BEFORE building
        # abs_pos. This mirrors IsaacLab `EMACumulativeDeltaPositionAction`
        # (see `actions.py:process_actions`): without this, repeated
        # high-entropy actions push `del_action` past the workspace, the EE
        # gets stuck at the boundary, and the policy can't recover (the EMA
        # clamp on `ema_pos` alone is insufficient because `del_action`
        # itself grows unbounded).
        if self._pos_lower is not None and self._pos_upper is not None:
            del_lower = self._pos_lower - self.init_ee_pos
            del_upper = self._pos_upper - self.init_ee_pos
            self.del_action = torch.clamp(self.del_action, del_lower, del_upper)

        # Absolute EE-position target in robot root frame.
        abs_pos = self.init_ee_pos + self.del_action

        # EMA smoothing.
        ema_pos = self._alpha * abs_pos + (1.0 - self._alpha) * self._prev_applied_pos

        # Defensive clamp on the final ema target (matches IsaacLab).
        if self._pos_lower is not None and self._pos_upper is not None:
            ema_pos = torch.clamp(ema_pos, self._pos_lower, self._pos_upper)

        # Build the 7-D pose target and drive the arm via parent's IK path.
        # We bypass `compute_target_pose` / `use_delta` since we already have
        # the absolute target.
        self._step = 0
        self._start_qpos = self.qpos
        self._target_pose = Pose.create_from_pq(ema_pos, self.init_ee_quat)
        self._target_qpos = self.kinematics.compute_ik(
            pose=self._target_pose,
            q0=self.articulation.get_qpos(),
            is_delta_pose=False,
            current_pose=self.ee_pose_at_base,
            solver_config=self.config.delta_solver_config,
        )
        if self._target_qpos is None:
            self._target_qpos = self._start_qpos
        if self.config.interpolate:
            self._step_size = (self._target_qpos - self._start_qpos) / self._sim_steps
        else:
            self.set_drive_targets(self._target_qpos)

        # Stash EMA pos for next step's EMA history.
        self._prev_applied_pos = ema_pos


@dataclass
class PDEEEMACumulativeDeltaPosControllerConfig(PDEEPosControllerConfig):
    """Config for the EMA-cumulative-delta EE-position controller.

    Mirrors IsaacLab ``EMACumulativeDeltaPositionActionCfg``: per-axis
    ``scale``, scalar ``alpha`` for EMA, optional per-axis position clamp.
    Action range is fixed at ``[-1, 1]^3`` (the policy's normalized space);
    ``pos_lower/pos_upper`` here are interpreted as the workspace clamp
    (``pos_lower_limit / pos_upper_limit`` in IsaacLab parlance).
    """

    scale: Union[float, Sequence[float]] = 0.01
    alpha: float = 0.5
    pos_lower_limit: Optional[Sequence[float]] = None
    pos_upper_limit: Optional[Sequence[float]] = None
    # Force ``use_delta=False`` and ``use_target=False`` — we manage state
    # ourselves and pass an absolute pose target to compute_ik.
    use_delta: bool = False
    use_target: bool = False
    normalize_action: bool = False    # action stays in raw [-1, 1] (no rescale)
    controller_cls = PDEEEMACumulativeDeltaPosController
