# tests/python/test_drone_nav_env.py

import sys
import pathlib
import numpy as np

HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[2]  # .../AeroForge

PYTHON_PACKAGE_DIR = ROOT / "python"
BINDINGS_DIR = ROOT / "build" / "src" / "bindings" / "python"

# Optional debug:
# print("ROOT:", ROOT)
# print("PYTHON_PACKAGE_DIR exists:", PYTHON_PACKAGE_DIR.exists())
# print("BINDINGS_DIR exists:", BINDINGS_DIR.exists())

sys.path.insert(0, str(PYTHON_PACKAGE_DIR))
sys.path.insert(0, str(BINDINGS_DIR))

from aeroforge.envs.drone_nav_env import DroneNavEnv




def test_full_thrust_increases_altitude():
    """With full thrust action [1,0,0,0], z should clearly increase."""

    env = DroneNavEnv(max_steps=200)
    obs, info = env.reset()

    z0 = float(obs[2])

    action = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    zs = [z0]
    steps = 50
    done = False

    for i in range(steps):
        obs, reward, done, info = env.step(action)
        zs.append(float(obs[2]))
        if done:
            break

    zf = zs[-1]

    # Basic sanity: final altitude should be noticeably higher than start
    assert zf > z0 + 0.2, f"Altitude did not increase enough: z0={z0}, zf={zf}"
