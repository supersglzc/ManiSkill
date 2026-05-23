"""
ManiSkill — random-action rollout that writes an MP4 (render sanity check).

Purpose
-------
Builds the benchmark env, runs N random-action steps, captures one RGB frame
per step via `env.render()` (SAPIEN offscreen GPU renderer), and writes them
to an MP4. Proves that the offscreen render pipeline is wired (camera obs /
env.render() / Vulkan stack) WITHOUT requiring a trained policy or X11.

Used as the L2 smoke tier by benchmark-generator. For headed visualization,
launch the script directly under your host display (the env's GUI window
will appear in parallel with the offscreen rendering).

Example
-------
    python scripts/render_random.py --task StackCube-v1 --n-steps 30 \\
        --output /tmp/ManiSkill_random.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    p.add_argument("--task", default="StackCube-v1",
                   help="Task identifier (default: %(default)s)")
    p.add_argument("--n-steps", type=int, default=30,
                   help="Frames to capture (default: %(default)s)")
    p.add_argument("--output", required=True,
                   help="Output MP4 path (e.g. /tmp/random.mp4)")
    p.add_argument("--fps", type=int, default=30,
                   help="MP4 frame rate (default: %(default)s)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_env(task: str):
    """Build the ManiSkill env with rgb_array rendering enabled.

    Mirrors `mani_skill/examples/demo_random_action.py` but pins
    `render_mode='rgb_array'` so `env.render()` returns an offscreen frame.
    """
    import mani_skill.envs  # noqa: F401 - registers @register_env tasks
    import gymnasium as gym
    env = gym.make(
        task,
        obs_mode="state",
        reward_mode="dense",
        control_mode=None,
        render_mode="rgb_array",
        num_envs=1,
        sim_backend="auto",
        render_backend="gpu",
    )
    return env


def sample_action(env):
    return env.action_space.sample()


def extract_frame(env, obs, info):
    """Return one RGB (H, W, 3) uint8 ndarray.

    ManiSkill returns a torch tensor of shape (num_envs, H, W, 3) uint8 from
    `env.render()`. We squeeze the leading env dim and move to CPU/numpy.
    """
    frame = env.render()
    if frame is None:
        return None
    # torch.Tensor or numpy.ndarray, both support .shape
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    return frame


def main():
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = build_env(args.task)
    try:
        obs, info = env.reset(seed=args.seed)
    except TypeError:
        obs = env.reset()
        info = {}
    except ValueError:
        # Older gym API returns just obs.
        obs = env.reset()
        info = {}

    frames: list[np.ndarray] = []
    for i in range(args.n_steps):
        out = env.step(sample_action(env))
        if isinstance(out, tuple) and len(out) == 5:
            obs, _r, _term, _trunc, info = out
        elif isinstance(out, tuple) and len(out) == 4:
            obs, _r, _done, info = out
        else:
            obs, info = out, {}
        frame = extract_frame(env, obs, info)
        if frame is not None:
            frames.append(np.asarray(frame, dtype=np.uint8))

    if not frames:
        print("FAIL: no RGB frames captured (check env.render() output)",
              file=sys.stderr)
        sys.exit(2)

    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio  # type: ignore[no-redef]

    imageio.mimsave(str(out_path), frames, fps=args.fps, codec="libx264", quality=8)
    size_kb = out_path.stat().st_size / 1024
    print(f"L2 OK: wrote {len(frames)} frames to {out_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
