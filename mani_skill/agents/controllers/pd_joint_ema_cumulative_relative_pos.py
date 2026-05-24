"""EMA cumulative-relative joint position controller for ManiSkill.

Direct port of IsaacLab's
``isaaclab_tasks.manager_based.manipulation.dex_grasp.mdp.actions.EMACumulativeRelativeJointPositionAction``
(itself vendored from bidex). Used by ``Isaac-Dex-Grasp-v1`` to match the
bidex/IsaacLab control loop semantics.

Per-step processing (raw action ``a``, scale ``s``, alpha ``α``, init joint
pose captured at reset ``q_init``):

    1. processed = s · a                                # JointPositionAction (offset disabled)
    2. processed += del_{t-1}                            # accumulate delta
    3. del_t      = processed.clone()                    # remember new cumulative delta
    4. processed += q_init                               # anchor on init pose
    5. ema        = α · processed + (1 − α) · prev_{t-1}
    6. processed  = clamp(ema, joint_lower_limit, joint_upper_limit)
    7. prev_t     = processed

State across steps (per env): ``del_action``, ``_prev_applied``,
``init_joint_pos``, ``_needs_reanchor``.

On ``reset(env_ids)``: ``del_action[env_ids] := 0``,
``_needs_reanchor[env_ids] := True`` — the NEXT ``set_action`` re-captures
``init_joint_pos`` and ``_prev_applied`` from the (now-fresh) qpos.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch

from mani_skill.utils.structs.types import Array

from .pd_joint_pos import PDJointPosController, PDJointPosControllerConfig


class PDJointEMACumulativeRelativePosController(PDJointPosController):
    """JointPos controller with per-episode EMA-smoothed cumulative-relative action.

    Action layout: ``[a_1, ..., a_dof]`` in ``[-1, 1]^dof`` (the env's
    ``action_space.low/high`` is fixed at ``[-1, 1]^dof``; the controller's
    ``cfg.lower/upper`` are interpreted as the post-EMA joint position clamp
    rather than per-step action limits).
    """

    config: "PDJointEMACumulativeRelativePosControllerConfig"

    def _initialize_joints(self):
        super()._initialize_joints()
        n_envs = self.scene.num_envs
        device = self.device
        dof = len(self.joints)
        self.del_action = torch.zeros((n_envs, dof), device=device)
        self.init_joint_pos = torch.zeros((n_envs, dof), device=device)
        self._prev_applied = torch.zeros((n_envs, dof), device=device)
        self._needs_reanchor = torch.ones(n_envs, dtype=torch.bool, device=device)
        # Per-joint position clamp (matches IsaacLab `joint_lower_limit/upper_limit`).
        if self.config.joint_lower_limit is not None and self.config.joint_upper_limit is not None:
            self._joint_lower = torch.tensor(self.config.joint_lower_limit, device=device, dtype=torch.float32)
            self._joint_upper = torch.tensor(self.config.joint_upper_limit, device=device, dtype=torch.float32)
        else:
            self._joint_lower = None
            self._joint_upper = None
        if not 0.0 <= self.config.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1]; got {self.config.alpha}")
        self._alpha = float(self.config.alpha)
        self._scale = float(self.config.scale)

    def _initialize_action_space(self):
        from gymnasium import spaces
        dof = len(self.joints) if hasattr(self, "joints") else self.action_dim
        low = np.full(dof, -1.0, dtype=np.float32)
        high = np.full(dof, 1.0, dtype=np.float32)
        self.single_action_space = spaces.Box(low, high, dtype=np.float32)

    def reset(self):
        super().reset()
        if not hasattr(self, "_needs_reanchor"):
            return
        partial = getattr(self, "_partial_reset_env_idx", None)
        if partial is not None:
            self._needs_reanchor[partial] = True
            self.del_action[partial] = 0.0
            self._partial_reset_env_idx = None
            return
        self._needs_reanchor[:] = True
        self.del_action[:] = 0.0

    def _preprocess_action(self, action: Array) -> torch.Tensor:
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, device=self.device, dtype=torch.float32)
        else:
            action = action.to(self.device).float()
        return action

    def set_action(self, action: Array):
        action = self._preprocess_action(action)

        # Re-anchor: capture current qpos as init + prev for reset envs.
        if self._needs_reanchor.any():
            cur_qpos = self.articulation.get_qpos()[:, self.active_joint_indices]
            mask = self._needs_reanchor
            self.init_joint_pos[mask] = cur_qpos[mask]
            self._prev_applied[mask] = cur_qpos[mask]
            self._needs_reanchor[:] = False

        # Step 1: scale · raw (no clamp on raw — IsaacLab doesn't either).
        processed = self._scale * action
        # Step 2 + 3: accumulate.
        processed = processed + self.del_action
        self.del_action = processed.clone()
        # Step 3.5: clamp the cumulative delta so `init + delta ∈ [lower, upper]`.
        # Without this `delta` keeps drifting past the limit boundary — the
        # absolute target still gets clipped each step, but `delta` remembers
        # the saturated direction so a reversing action has to burn through the
        # accumulated overshoot before any joint motion resumes. Matches the
        # IsaacLab template (templates/task-generator/action_terms/ema_delta_joint_pos.py).
        if self._joint_lower is not None and self._joint_upper is not None:
            del_lower = self._joint_lower - self.init_joint_pos
            del_upper = self._joint_upper - self.init_joint_pos
            self.del_action = torch.clamp(self.del_action, del_lower, del_upper)
            processed = self.del_action
        # Step 4: anchor on init joint pose.
        processed = processed + self.init_joint_pos
        # Step 5: EMA against prev.
        ema = self._alpha * processed + (1.0 - self._alpha) * self._prev_applied
        # Step 6: clamp.
        if self._joint_lower is not None and self._joint_upper is not None:
            target = torch.clamp(ema, self._joint_lower, self._joint_upper)
        else:
            target = ema
        # Step 7: update prev for next step.
        self._prev_applied = target

        # Drive the joint position target via the parent's set_drive_targets.
        self._step = 0
        self._start_qpos = self.qpos
        self._target_qpos = target
        if self.config.interpolate:
            self._step_size = (self._target_qpos - self._start_qpos) / self._sim_steps
        else:
            self.set_drive_targets(self._target_qpos)


@dataclass
class PDJointEMACumulativeRelativePosControllerConfig(PDJointPosControllerConfig):
    """Config for the EMA cumulative-relative joint position controller.

    Mirrors IsaacLab ``EMACumulativeRelativeJointPositionActionCfg``:
    scalar ``scale``, scalar ``alpha`` for EMA, and explicit per-joint
    position-clamp lists (``joint_lower_limit / joint_upper_limit``).
    """
    scale: float = 0.03
    alpha: float = 0.2
    joint_lower_limit: Optional[Sequence[float]] = None
    joint_upper_limit: Optional[Sequence[float]] = None
    normalize_action: bool = False    # action stays in raw [-1, 1] — we apply scale ourselves
    controller_cls = PDJointEMACumulativeRelativePosController
