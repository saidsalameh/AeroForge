# tests/python/test_drone_nav_env.py

import sys
import pathlib
import numpy as np

HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[2]  # .../AeroForge

PYTHON_PACKAGE_DIR = ROOT / "python"
BINDINGS_DIR = ROOT / "build" / "src" / "bindings" / "python"

sys.path.insert(0, str(PYTHON_PACKAGE_DIR))
sys.path.insert(0, str(BINDINGS_DIR))

from aeroforge.envs.drone_nav_env import DroneNavEnv


# def test_full_thrust_increases_altitude():
#     """With full thrust action [1,0,0,0], altitude z should clearly increase."""

#     env = DroneNavEnv(max_steps=200)
#     obs, info = env.reset()

#     # obs layout: [z, vz, dz, roll, pitch, p, q, r]
#     z0 = float(obs[0])

#     action = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

#     zs = [z0]
#     steps = 50
#     done = False

#     for i in range(steps):
#         obs, reward, done, info = env.step(action)
#         z = float(obs[0])  # current altitude
#         zs.append(z)
#         if done:
#             break

#     zf = zs[-1]

    # Basic sanity: final altitude should be noticeably higher than start
    # assert zf > z0 + 0.2, f"Altitude did not increase enough: z0={z0}, zf={zf}"


def test_hover_reward_at_target():
    """At target altitude with zero motion and level attitude, reward should be near zero."""

    env = DroneNavEnv(max_steps=100)
    obs, info = env.reset()

    target_z = float(env.target_position[2])

    # obs layout: [z, vz, dz, roll, pitch, p, q, r]
    # Hover at target: z = target_z, vz = 0, dz = 0, roll = pitch = p = q = r = 0
    hover_obs = np.array(
        [target_z, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )

    reward = env.compute_reward(hover_obs)

    # With no errors and no motion, reward should be close to 0
    assert abs(reward) < 1e-3, f"Hover reward not near zero: reward={reward}"


# def test_episode_ends_on_max_steps():
#     """Episode should end after max_steps steps."""

#     max_steps = 10
#     env = DroneNavEnv(max_steps=max_steps)
#     obs, info = env.reset()

#     action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

#     done = False
#     for step in range(max_steps):
#         obs, reward, done, info = env.step(action)
#         if step < max_steps - 1:
#             assert not done, f"Episode ended prematurely at step {step}"
#         else:
#             assert done, f"Episode did not end at max_steps {max_steps}"
