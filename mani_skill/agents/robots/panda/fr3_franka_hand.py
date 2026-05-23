"""FR3 (Franka Research 3) + Franka-hand agent — mirrors the robot used by
IsaacLab's `Triton-Franka-StackCube` task. URDF lives at
`<repo>/nautilus/assets/fr3/fr3_franka_hand.urdf` (preprocessed from
`/home/steven/code/agentic/franka_description/urdfs/fr3_franka_hand.urdf`
to replace `package://franka_description/` paths with absolute filesystem
paths so SAPIEN's URDFLoader can resolve mesh references).

Joint layout (DOF=9):
    fr3_joint1..fr3_joint7  — revolute arm joints
    fr3_finger_joint1/2     — prismatic gripper fingers (0..0.04 m)

TCP link: `fr3_hand_tcp` (offset `(0, 0, 0.209)` from `fr3_hand`).
"""
from copy import deepcopy
from pathlib import Path

import numpy as np
import sapien
import torch

from mani_skill.agents.base_agent import Keyframe
from mani_skill.agents.controllers import (
    PDEEEMACumulativeDeltaPosControllerConfig,
    PDEEPosControllerConfig,
    PDEEPoseControllerConfig,
    PDJointPosControllerConfig,
    PDJointPosMimicControllerConfig,
)
from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.utils import sapien_utils


# Absolute URDF path. Preprocessed copy lives inside the ManiSkill repo so the
# URDF is portable across `pip install -e .` builds. Path structure:
#   <repo>/mani_skill/agents/robots/panda/fr3_franka_hand.py
#   parents[0]=panda  [1]=robots  [2]=agents  [3]=mani_skill  [4]=<repo>
_FR3_URDF_PATH = str(
    Path(__file__).resolve().parents[4]
    / "nautilus" / "assets" / "fr3" / "fr3_franka_hand.urdf"
)


@register_agent()
class FR3FrankaHand(Panda):
    """FR3 + Franka-hand — Panda subclass with FR3 joint/link names and
    IsaacLab-matched init joint pose.

    Differences from stock `panda` / `panda_wristcam`:
      * URDF: FR3 (longer link4/5, different inertia) + Franka-hand gripper
      * Joint name prefix: `fr3_*` instead of `panda_*`
      * TCP link: `fr3_hand_tcp`
      * Init keyframe (`rest`): mirrors IsaacLab `FRANKA_INIT_JOINT_POS`
      * `_controller_configs` overrides `pd_ee_delta_pos` and
        `pd_ee_target_delta_pos` to use `pos_lower=-0.01, pos_upper=0.01` so
        the effective per-step EE delta matches IsaacLab `scale=0.01`.
    """

    uid = "fr3_franka_hand"
    urdf_path = _FR3_URDF_PATH

    # Friction overrides apply to fr3 finger links instead of panda_*.
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            fr3_leftfinger=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
            fr3_rightfinger=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
        ),
    )

    # IsaacLab `FRANKA_INIT_JOINT_POS` — joint1=-0.785 rotates arm 45° toward
    # the cube spawn region; joint2/4/6 set a lowered EE arc; fingers open.
    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [-0.785, -0.785, 0.0, -2.655, 0.0, 1.87, 0.0, 0.04, 0.04]
            ),
            pose=sapien.Pose(),
        )
    )

    arm_joint_names = [
        "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
        "fr3_joint5", "fr3_joint6", "fr3_joint7",
    ]
    gripper_joint_names = ["fr3_finger_joint1", "fr3_finger_joint2"]
    ee_link_name = "fr3_hand_tcp"

    # Stiffness / damping inherited from Panda (1e3 / 1e2 / 100 N·m on arm and
    # gripper). IsaacLab uses higher arm stiffness (400 on shoulder/forearm,
    # 2e3 on hand actuators), but the Panda numbers are tracked-target-stable
    # under ManiSkill's IK at the timesteps we use.

    @property
    def _controller_configs(self):
        configs = super()._controller_configs
        # Tighten EE-delta limits to match IsaacLab `scale=0.01` per-step
        # absolute delta. ManiSkill default for `pd_ee_delta_pos` is ±0.1 m;
        # we want ±0.01 m for an effective-delta match with IsaacLab.
        for ctrl_name in ("pd_ee_delta_pos", "pd_ee_target_delta_pos"):
            if ctrl_name in configs:
                configs[ctrl_name]["arm"].pos_lower = -0.01
                configs[ctrl_name]["arm"].pos_upper = 0.01
        # Re-target the gripper mimic dict from `panda_finger_joint*` to
        # `fr3_finger_joint*` (parent class hard-codes panda names at
        # panda.py:184). The mimic shape is unchanged — one control joint
        # commands the other.
        for ctrl_name, ctrl_cfg in configs.items():
            if "gripper" in ctrl_cfg and getattr(ctrl_cfg["gripper"], "mimic", None):
                ctrl_cfg["gripper"].mimic = {
                    "fr3_finger_joint2": {"joint": "fr3_finger_joint1"}
                }

        # Add the EMA cumulative-delta EE-position controller — direct port
        # of IsaacLab's ``EMACumulativeDeltaPositionAction`` (xyz only, RPY
        # locked, scale=0.01, alpha=0.5). Mirrors the IsaacLab
        # ``Triton-Franka-StackCube`` task's §2 action.
        arm_pd_ee_ema_delta_pos = PDEEEMACumulativeDeltaPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.01,                # nominal; not used (normalize_action=False)
            pos_upper=0.01,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            scale=0.01,                     # IsaacLab scale
            alpha=0.5,                      # IsaacLab alpha
            # Skip workspace clamp here — IsaacLab uses pos_lower_limit/
            # pos_upper_limit in robot root frame; matching exactly is
            # secondary and we leave them None.
            pos_lower_limit=None,
            pos_upper_limit=None,
        )
        gripper = configs["pd_ee_delta_pos"]["gripper"]  # reuse Panda's gripper cfg
        configs["pd_ee_ema_delta_pos"] = dict(arm=arm_pd_ee_ema_delta_pos, gripper=gripper)
        return configs

    def _after_init(self):
        # Override link references to use fr3_* names.
        self.finger1_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "fr3_leftfinger"
        )
        self.finger2_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "fr3_rightfinger"
        )
        # FR3 URDF does not define separate finger-pad links — point both pad
        # references at the finger links so downstream `is_grasping`/contact
        # checks behave the same.
        self.finger1pad_link = self.finger1_link
        self.finger2pad_link = self.finger2_link
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )
