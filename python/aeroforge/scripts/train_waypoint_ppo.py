import os
import sys
import pathlib

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# ---------------------------------------------------------------------------
# Path setup so we can import aeroforge + compiled bindings
# ---------------------------------------------------------------------------

HERE = pathlib.Path(__file__).resolve()
# Expected path: python/aeroforge/scripts/train_waypoint_ppo.py
ROOT = HERE.parents[3]

PYTHON_PACKAGE_DIR = ROOT / "python"
BINDINGS_DIR = ROOT / "build" / "src" / "bindings" / "python"

LOG_PATH = ROOT / "tests" / "logs" / "train_waypoint.txt"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

sys.stdout = open(LOG_PATH, "w")
sys.stderr = sys.stdout

sys.path.insert(0, str(PYTHON_PACKAGE_DIR))
sys.path.insert(0, str(BINDINGS_DIR))

from aeroforge.envs.DroneWaypointEnv import DroneWaypointEnv


# ---------------------------------------------------------------------------
# Gymnasium-compatible wrapper for DroneWaypointEnv
# ---------------------------------------------------------------------------

class DroneWaypointGymEnv(gym.Env):
    """
    Adapter to make DroneWaypointEnv usable by Stable-Baselines3.

    DroneWaypointEnv API:
        reset() -> (obs, info)
        step(action) -> (obs, reward, done, info)

    Gymnasium API:
        reset(seed=None, options=None) -> (obs, info)
        step(action) -> (obs, reward, terminated, truncated, info)
    """

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 1000, waypoints=None):
        super().__init__()
        self._env = DroneWaypointEnv(max_steps=max_steps, waypoints=waypoints)

        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        obs, info = self._env.reset()
        return obs, info

    def step(self, action):
        obs, reward, done, info = self._env.step(action)

        terminated = bool(done)
        truncated = False  # we encode time-limit inside env as termination

        return obs, reward, terminated, truncated, info

    def close(self):
        self._env.close()
        super().close()

    @property
    def step_count(self):
        return self._env.step_count


# ---------------------------------------------------------------------------
# Env factory for vectorization
# ---------------------------------------------------------------------------

def make_env(max_steps: int = 1000, waypoints=None):
    """Factory used by DummyVecEnv to create multiple env instances."""
    def _init():
        return DroneWaypointGymEnv(max_steps=max_steps, waypoints=waypoints)
    return _init


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

N_ENVS = 4
TOTAL_TIMESTEPS = 300_000

LEARNING_RATE = 3e-4
N_STEPS = 1024
BATCH_SIZE = 256
GAMMA = 0.99

HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[3]

LOG_DIR = ROOT / "logs" / "waypoint_ppo"
MODEL_DIR = ROOT / "models" / "waypoint_ppo"
MODEL_PATH = MODEL_DIR / "ppo_drone_waypoints"


def main():
    # Create dirs
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Example waypoint pattern (same as default)
    waypoints = np.array(
        [
            [2.0, 2.0, 1.0],
            [2.0, -2.0, 1.0],
            [-2.0, -2.0, 1.0],
            [-2.0, 2.0, 1.0],
        ],
        dtype=np.float64,
    )

    # -----------------------------------------------------------------------
    # 1) Vectorized environment
    # -----------------------------------------------------------------------
    env_fns = [make_env(max_steps=1000, waypoints=waypoints) for _ in range(N_ENVS)]
    vec_env = DummyVecEnv(env_fns)

    # -----------------------------------------------------------------------
    # 2) PPO model
    # -----------------------------------------------------------------------
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        verbose=1,
        tensorboard_log=str(LOG_DIR),
    )

    # -----------------------------------------------------------------------
    # 3) Train
    # -----------------------------------------------------------------------
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # -----------------------------------------------------------------------
    # 4) Save model
    # -----------------------------------------------------------------------
    model.save(str(MODEL_PATH))
    print(f"Saved waypoint PPO model to: {MODEL_PATH}")

    vec_env.close()

    # -----------------------------------------------------------------------
    # 5) Quick evaluation on a single env
    # -----------------------------------------------------------------------
    eval_env = DroneWaypointGymEnv(max_steps=1000, waypoints=waypoints)
    model = PPO.load(str(MODEL_PATH))

    n_eval_episodes = 3

    for ep in range(n_eval_episodes):
        obs, info = eval_env.reset()
        done = False
        ep_reward = 0.0

        print("\n" + "=" * 70)
        print(f"[Eval] Episode {ep+1}/{n_eval_episodes}")
        print(f"Initial info: {info}")
        print("=" * 70)

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            ep_reward += reward

            x, y, z = obs[0:3]
            dx, dy, dz = obs[12:15]
            d = np.sqrt(dx * dx + dy * dy + dz * dz)

            print(f"[Eval] Step {eval_env.step_count}")
            print(f"  • pos (x,y,z)            : {x: .3f}, {y: .3f}, {z: .3f}")
            print(f"  • err (dx,dy,dz)         : {dx: .3f}, {dy: .3f}, {dz: .3f}")
            print(f"  • dist to current WP     : {d: .3f}")
            print(f"  • reward                 : {reward: .3f}")
            print(f"  • cum. reward            : {ep_reward: .3f}")
            print(f"  • wp index               : {info.get('current_waypoint_index')}")
            print(f"  • waypoint_reached       : {info.get('waypoint_reached')}")
            print(f"  • mission_complete       : {info.get('mission_complete')}")
            print("-" * 70)

        print(f"[Eval] Episode {ep+1} finished, total reward = {ep_reward: .3f}")

    eval_env.close()


if __name__ == "__main__":
    main()
