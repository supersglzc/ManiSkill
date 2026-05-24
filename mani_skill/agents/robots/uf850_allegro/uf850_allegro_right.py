"""UF850 + Allegro right-hand agent for ManiSkill — mirrors IsaacLab's
``Isaac-Dex-Grasp`` task (bidex right-robot variant).

22-DoF: 6 arm joints (``joint1..joint6``) + 16 Allegro right-hand joints
(``j{if,mf,pf,th}{1..4}``). Init joint pose, per-joint actuator stiffness /
damping groups, and joint-limit ranges all match the IsaacLab spec at
``IsaacLab/source/.../manager_based/manipulation/dex_grasp/config/uf850/joint_pos_env_cfg.py``.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np
import sapien
import torch

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import (
    PDJointEMACumulativeRelativePosControllerConfig,
    PDJointPosControllerConfig,
)
from mani_skill.agents.registration import register_agent
from mani_skill.utils import sapien_utils


# Mirror IsaacLab `JOINT_LOWER_LIMIT / JOINT_UPPER_LIMIT` in `actions_cfg.py`.
# Joint order matches the URDF active-joint enumeration:
#   joint1..joint6 (arm)
#   jif1, jmf1, jpf1, jth1   (proximal)
#   jif2, jmf2, jpf2, jth2   (mid)
#   jif3, jmf3, jpf3, jth3   (distal)
#   jif4, jmf4, jpf4, jth4   (tip)
JOINT_LOWER_LIMIT = [
    -6.283, -2.304, -4.224, -6.283, -2.164, -6.283,
    -0.05,  -0.05,  -0.570,  0.364,
    -0.296, -0.296, -0.296, -0.205,
    -0.274, -0.274, -0.274, -0.290,
    -0.327, -0.327, -0.327, -0.262,
]
JOINT_UPPER_LIMIT = [
     6.283,  2.304,  0.061,  6.283,  2.164,  6.283,
     0.570,  0.05,   0.05,   1.497,
     1.710,  1.710,  1.710,  1.130,
     1.809,  1.809,  1.809,  1.633,
     1.718,  1.718,  1.718,  1.820,
]

# Init joint pose — bidex right-robot verbatim (matches IsaacLab `ROBOT_INIT_JOINT_POS`).
# URDF joint order: arm joint1..6, then jif1,jmf1,jpf1,jth1, jif2,jmf2,jpf2,jth2, ...
INIT_QPOS = np.array(
    [
        # joint1..joint6 (arm)
        0.8, 0.3, -0.6, 0.0, -0.8, -1.57,
        # f1 (proximal)  jif1, jmf1, jpf1, jth1
        0.0, 0.0, 0.0, 1.3,
        # f2 (mid)       jif2, jmf2, jpf2, jth2
        0.4, 0.4, 0.4, 0.0,
        # f3 (distal)    jif3, jmf3, jpf3, jth3
        0.4, 0.4, 0.4, 0.2,
        # f4 (tip)       jif4, jmf4, jpf4, jth4
        0.0, 0.0, 0.0, 0.0,
    ],
    dtype=np.float32,
)


_URDF_PATH = str(
    Path(__file__).resolve().parents[4]
    / "nautilus" / "assets" / "uf850_allegro" / "uf850_allegro_right.urdf"
)


@register_agent()
class UF850AllegroRight(BaseAgent):
    """UF850 6-DoF arm + Allegro right hand (16-DoF) — 22-DoF total."""

    uid = "uf850_allegro_right"
    urdf_path = _URDF_PATH
    # No fingertip friction / patch-radius overrides — contact-related material
    # tweaks were removed per user request to keep the env's contact constraints
    # purely physics-default. Reward shaping no longer references contact forces.
    urdf_config = dict()

    keyframes = dict(
        rest=Keyframe(qpos=INIT_QPOS.copy(), pose=sapien.Pose()),
    )

    arm_joint_names = [
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
    ]
    # Hand joints in URDF active-order. (proximal of 4 fingers, then mid, distal, tip)
    hand_joint_names = [
        "jif1", "jmf1", "jpf1", "jth1",
        "jif2", "jmf2", "jpf2", "jth2",
        "jif3", "jmf3", "jpf3", "jth3",
        "jif4", "jmf4", "jpf4", "jth4",
    ]
    all_joint_names = arm_joint_names + hand_joint_names
    ee_link_name = "palm_link"

    # Per-joint actuator gains — match IsaacLab `ImplicitActuatorCfg` groups in
    # `uf850/joint_pos_env_cfg.py:UF850_ALLEGRO_RIGHT_CFG.actuators`.
    @property
    def _controller_configs(self):
        # Map joint_name -> (stiffness, damping)
        gains: dict[str, tuple[float, float]] = {}
        # arm joints (xArm_1-6 group)
        for j in self.arm_joint_names:
            gains[j] = (2000.0, 16.0)
        # hand fingers (4 groups by suffix f1/f2/f3/f4 — for index/middle/pinky)
        # and per-joint thumb groups (jth1..jth4).
        f_group = {
            "f1": (325.0, 20.0),
            "f2": (425.0, 25.0),
            "f3": (245.0, 15.0),
            "f4": (1050.0, 65.0),
        }
        thumb_group = {
            "jth1": (100.0, 5.0),
            "jth2": (300.0, 15.0),
            "jth3": (1270.0, 100.0),
            "jth4": (1000.0, 50.0),
        }
        for j in self.hand_joint_names:
            if j.startswith("jth"):
                gains[j] = thumb_group[j]
            else:
                # j{if,mf,pf}{1..4}
                suffix = j[-2:]   # "f1" / "f2" / "f3" / "f4"
                gains[j] = f_group[suffix]
        # Build per-joint stiffness / damping arrays in URDF order.
        stiff = [gains[j][0] for j in self.all_joint_names]
        damp  = [gains[j][1] for j in self.all_joint_names]

        # EMA cumulative-relative joint pos — matches IsaacLab spec exactly.
        ema_joint_pos = PDJointEMACumulativeRelativePosControllerConfig(
            joint_names=self.all_joint_names,
            lower=None,
            upper=None,
            stiffness=stiff,
            damping=damp,
            force_limit=1e10,
            scale=0.05,
            alpha=0.2,
            joint_lower_limit=JOINT_LOWER_LIMIT,
            joint_upper_limit=JOINT_UPPER_LIMIT,
        )
        # Plain PD joint pos (passthrough) — useful for eval / scripted tests.
        plain_joint_pos = PDJointPosControllerConfig(
            joint_names=self.all_joint_names,
            lower=JOINT_LOWER_LIMIT,
            upper=JOINT_UPPER_LIMIT,
            stiffness=stiff,
            damping=damp,
            force_limit=1e10,
            normalize_action=False,
        )
        return dict(
            ema_joint_pos=dict(arm_hand=ema_joint_pos),
            pd_joint_pos=dict(arm_hand=plain_joint_pos),
        )

    def _after_init(self):
        self.palm_link = sapien_utils.get_obj_by_name(self.robot.get_links(), "palm_link")
        self.if5_link = sapien_utils.get_obj_by_name(self.robot.get_links(), "if5")
        self.mf5_link = sapien_utils.get_obj_by_name(self.robot.get_links(), "mf5")
        self.pf5_link = sapien_utils.get_obj_by_name(self.robot.get_links(), "pf5")
        self.th5_link = sapien_utils.get_obj_by_name(self.robot.get_links(), "th5")
        # Generic alias so downstream code that expects `.tcp` works.
        self.tcp = self.palm_link
