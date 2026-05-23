"""
ManiSkill — random-action rollout (env sanity check).

Purpose
-------
Builds the benchmark env, runs N random-action steps, and verifies that
`env.step(...)` returns a finite numeric reward on every step. This is the
fastest end-to-end check that env-generator + benchmark-generator wired the
simulator correctly. NO trained policy is needed; NO video is produced.

Used as the L1 smoke tier by benchmark-generator and as the day-1 sanity
check a user can run after `bash nautilus/setup_uv.sh`.

Example
-------
    python scripts/run_random.py --task StackCube-v1 --n-steps 10
    python scripts/run_random.py --task StackCube-v1 --n-steps 100 --seed 0
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    p.add_argument("--task", default="StackCube-v1",
                   help="Task identifier (default: %(default)s)")
    p.add_argument("--n-steps", type=int, default=10,
                   help="Number of env.step() calls (default: %(default)s)")
    p.add_argument("--seed", type=int, default=0,
                   help="env.reset(seed=...) when supported (default: %(default)s)")
    return p.parse_args()


def build_env(task: str):
    """Build the ManiSkill env headless (no render).

    Mirrors `mani_skill/examples/demo_random_action.py`:
    `gym.make(env_id, obs_mode='state', reward_mode='dense', num_envs=1,
              sim_backend='auto', render_mode=None)`.
    """
    import mani_skill.envs  # noqa: F401 - registers @register_env tasks
    import gymnasium as gym
    env = gym.make(
        task,
        obs_mode="state",
        reward_mode="dense",
        control_mode=None,
        render_mode=None,
        num_envs=1,
        sim_backend="auto",
    )
    return env


def sample_action(env):
    """Random action — gym Box sample, shape matches the env's action_space."""
    return env.action_space.sample()


def main():
    args = parse_args()
    env = build_env(args.task)
    try:
        env.reset(seed=args.seed)
    except TypeError:
        env.reset()

    bad = 0
    for i in range(args.n_steps):
        out = env.step(sample_action(env))
        # Support 4-tuple (gym) and 5-tuple (gymnasium) and dm_env-style TimeStep.
        reward = out[1] if isinstance(out, tuple) else getattr(out, "reward", None)
        # ManiSkill returns a 1-element torch tensor; collapse to a python float.
        try:
            reward = float(reward)
        except (TypeError, ValueError):
            pass
        if reward is None or not np.isfinite(reward):
            print(f"step {i}: reward={reward!r} (NOT finite)", file=sys.stderr)
            bad += 1
        else:
            print(f"step {i}: reward={float(reward):.4f}")

    if bad:
        print(f"FAIL: {bad}/{args.n_steps} steps returned non-finite reward", file=sys.stderr)
        sys.exit(1)
    print(f"L1 OK: {args.n_steps} steps, all rewards finite")


if __name__ == "__main__":
    main()
