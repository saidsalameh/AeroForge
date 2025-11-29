# tests/python/test_drone_nav_env.py

import sys
import pathlib
import numpy as np

# ---------------------------------------------------------------------
# Path setup: make sure we can import the Python package + bindings
# ---------------------------------------------------------------------

HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[2]  # .../AeroForge

PYTHON_PACKAGE_DIR = ROOT / "python"
BINDINGS_DIR = ROOT / "build" / "src" / "bindings" / "python"

sys.path.insert(0, str(PYTHON_PACKAGE_DIR))
sys.path.insert(0, str(BINDINGS_DIR))

from aeroforge.envs.drone_nav_env import DroneNavEnv


# ---------------------------------------------------------------------
# Helper to build a consistent 15D observation for Stage 5
#
# Layout in DroneNavEnv (Stage 5):
#   obs = [
#       x, y, z,
#       vx, vy, vz,
#       roll, pitch, yaw,
#       p, q, r,
#       dx, dy, dz
#   ]
# where dx = x - target_x, dy = y - target_y, dz = z - target_z
# ---------------------------------------------------------------------
def make_obs(x, y, z,
             vx, vy, vz,
             roll, pitch, yaw,
             p, q, r,
             target_pos):
    tx, ty, tz = target_pos
    dx = x - tx
    dy = y - ty
    dz = z - tz
    return np.array(
        [x, y, z,
         vx, vy, vz,
         roll, pitch, yaw,
         p, q, r,
         dx, dy, dz],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------
# 1) Shape & type test
# ---------------------------------------------------------------------
def test_reset_and_step_shape_and_types():
    """reset() and step() must return obs with shape (15,) and proper types."""
    env = DroneNavEnv(max_steps=20)
    obs, info = env.reset()

    # Shape of observation
    assert obs.shape == (15,), f"reset() obs shape is {obs.shape}, expected (15,)"

    # Info is a dict
    assert isinstance(info, dict), f"reset() info should be dict, got {type(info)}"

    # One step with zero action
    action = np.zeros(4, dtype=np.float64)
    obs2, reward, done, info2 = env.step(action)

    # Shape still (15,)
    assert obs2.shape == (15,), f"step() obs shape is {obs2.shape}, expected (15,)"

    # All finite
    assert np.all(np.isfinite(obs2)), "step() observation contains non-finite values"

    # Types of reward and done
    assert isinstance(reward, (float, np.floating)), f"reward has type {type(reward)}"
    assert isinstance(done, bool), f"done has type {type(done)}, expected bool"
    assert isinstance(info2, dict), f"info has type {type(info2)}, expected dict"


# ---------------------------------------------------------------------
# 2) Distance-based reward sanity
# ---------------------------------------------------------------------
def test_reward_better_when_closer_to_target():
    """Reward should be higher (less negative) when the drone is closer to the target."""

    env = DroneNavEnv(max_steps=100)
    obs0, _ = env.reset()
    target = env.target_position.copy()

    # Far from target (e.g. +5m in x,y,z)
    obs_far = make_obs(
        x=target[0] + 5.0,
        y=target[1] + 5.0,
        z=target[2] + 5.0,
        vx=0.0, vy=0.0, vz=0.0,
        roll=0.0, pitch=0.0, yaw=0.0,
        p=0.0, q=0.0, r=0.0,
        target_pos=target,
    )

    # Closer to target (e.g. +0.5m in x,y,z)
    obs_close = make_obs(
        x=target[0] + 0.5,
        y=target[1] + 0.5,
        z=target[2] + 0.5,
        vx=0.0, vy=0.0, vz=0.0,
        roll=0.0, pitch=0.0, yaw=0.0,
        p=0.0, q=0.0, r=0.0,
        target_pos=target,
    )

    # Ignore progress term by resetting last_distance before each call
    env.last_distance = None
    r_far = env.compute_reward(obs_far)

    env.last_distance = None
    r_close = env.compute_reward(obs_close)

    assert r_close > r_far, (
        f"Reward for closer state should be higher. "
        f"r_far={r_far}, r_close={r_close}"
    )


# ---------------------------------------------------------------------
# 3) Progress reward monotonicity
# ---------------------------------------------------------------------
def test_progress_reward_improves_when_distance_decreases():
    """
    With the same last_distance, a state closer to the target should get
    a larger progress term contribution than a farther state.
    """

    env = DroneNavEnv(max_steps=100)
    obs0, _ = env.reset()
    target = env.target_position.copy()

    # Two states with different distances
    obs_far = make_obs(
        x=target[0] + 3.0,
        y=target[1] + 0.0,
        z=target[2] + 0.0,
        vx=0.0, vy=0.0, vz=0.0,
        roll=0.0, pitch=0.0, yaw=0.0,
        p=0.0, q=0.0, r=0.0,
        target_pos=target,
    )
    obs_closer = make_obs(
        x=target[0] + 1.0,
        y=target[1] + 0.0,
        z=target[2] + 0.0,
        vx=0.0, vy=0.0, vz=0.0,
        roll=0.0, pitch=0.0, yaw=0.0,
        p=0.0, q=0.0, r=0.0,
        target_pos=target,
    )

    # Set the same last_distance artificially to isolate the progress term behavior
    env.last_distance = 5.0
    r_far = env.compute_reward(obs_far)

    env.last_distance = 5.0
    r_closer = env.compute_reward(obs_closer)

    assert r_closer > r_far, (
        "Progress reward should make the closer state less penalized. "
        f"r_far={r_far}, r_closer={r_closer}"
    )


# ---------------------------------------------------------------------
# 4) Episode termination behavior (time limit + in-bounds sanity)
# ---------------------------------------------------------------------
def test_episode_ends_on_max_steps():
    """Episode should end after max_steps steps due to time limit."""

    max_steps = 5
    env = DroneNavEnv(max_steps=max_steps)
    obs, info = env.reset()

    action = np.zeros(4, dtype=np.float64)

    done = False
    for step in range(max_steps):
        obs, reward, done, info = env.step(action)
        if step < max_steps - 1:
            assert not done, f"Episode ended prematurely at step {step}"
        else:
            assert done, f"Episode did not end at max_steps={max_steps}"


def test_check_done_in_bounds_is_false():
    """check_done() should return False for a safe, in-bounds state."""

    env = DroneNavEnv(max_steps=100)
    obs0, _ = env.reset()
    target = env.target_position.copy()

    # State close to target, zero velocity, small angles
    obs_safe = make_obs(
        x=target[0] + 0.1,
        y=target[1] - 0.1,
        z=target[2] + 0.05,
        vx=0.0, vy=0.0, vz=0.0,
        roll=0.05, pitch=-0.05, yaw=0.0,
        p=0.0, q=0.0, r=0.0,
        target_pos=target,
    )

    done = env.check_done(obs_safe)
    assert done is False, "Safe in-bounds state should not terminate the episode"
