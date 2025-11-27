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
# This file is expected at: python/aeroforge/scripts/train_hover_ppo.py
# parents[0] -> scripts
# parents[1] -> aeroforge
# parents[2] -> python
# parents[3] -> AeroForge (project root)
ROOT = HERE.parents[3]

PYTHON_PACKAGE_DIR = ROOT / "python"
BINDINGS_DIR = ROOT / "build" / "src" / "bindings" / "python"

sys.path.insert(0, str(PYTHON_PACKAGE_DIR))
sys.path.insert(0, str(BINDINGS_DIR))

from aeroforge.envs.drone_nav_env import DroneNavEnv


# ---------------------------------------------------------------------------
# Gymnasium-compatible wrapper around DroneNavEnv
# ---------------------------------------------------------------------------

class DroneNavGymEnv(gym.Env):
    """
    Small adapter to make DroneNavEnv compatible with Gymnasium / Stable-Baselines3.

    Internally uses DroneNavEnv, which has:
        reset() -> (obs, info)
        step(action) -> (obs, reward, done, info)

    Here we expose:
        reset(seed=None, options=None) -> (obs, info)
        step(action) -> (obs, reward, terminated, truncated, info)
    """

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 500):
        super().__init__()
        self._env = DroneNavEnv(max_steps=max_steps)

        # Reuse spaces from the underlying environment
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        obs, info = self._env.reset()
        return obs, info

    def step(self, action):
        # DroneNavEnv returns: obs, reward, done, info
        obs, reward, done, info = self._env.step(action)

        # Gymnasium separates termination/reason:
        terminated = bool(done)   # episode ended due to task or failure
        truncated = False         # no time-limit truncation handled here (env does it as termination)

        return obs, reward, terminated, truncated, info

    def close(self):
        self._env.close()
        super().close()

    @property
    def step_count(self):
        return self._env.step_count


# ---------------------------------------------------------------------------
# Environment factory for vectorization
# ---------------------------------------------------------------------------

def make_env(max_steps: int = 500):
    """Factory that returns a function creating a fresh DroneNavGymEnv."""
    def _init():
        return DroneNavGymEnv(max_steps=max_steps)
    return _init


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

N_ENVS = 4
TOTAL_TIMESTEPS = 100_000

LEARNING_RATE = 3e-4
N_STEPS = 1024
BATCH_SIZE = 256
GAMMA = 0.99

LOG_DIR = ROOT / "logs" / "hover_ppo"
MODEL_DIR = ROOT / "models" / "hover_ppo"
MODEL_PATH = MODEL_DIR / "ppo_drone_hover"


def main():
    # Create output directories if they don't exist
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1) Create vectorized environment
    # -----------------------------------------------------------------------
    env_fns = [make_env(max_steps=500) for _ in range(N_ENVS)]
    vec_env = DummyVecEnv(env_fns)

    # -----------------------------------------------------------------------
    # 2) Instantiate PPO model
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
    print(f"Saved trained model to: {MODEL_PATH}")

    # Optional: free resources
    vec_env.close()

    # -----------------------------------------------------------------------
    # 5) Quick evaluation run (single env, deterministic policy)
    # -----------------------------------------------------------------------
    eval_env = DroneNavGymEnv(max_steps=500)
    model = PPO.load(str(MODEL_PATH))

    n_eval_episodes = 5
    for ep in range(n_eval_episodes):
        obs, info = eval_env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            ep_reward += reward

            print("\n" + "============================================================")
            print(f"[Eval] Episode {ep+1} | Step {eval_env.step_count}")
            print("-------------------------------------------------------------")


            print(f"  • Reward this step     : {reward: .4f}")
            print(f"  • Cumulative reward    : {ep_reward: .4f}")

            print("\n  • Observation:")
            print(f"       z (altitude)      : {obs[0]: .4f}")
            print(f"       vz (vert. vel)    : {obs[1]: .4f}")
            print(f"       dz (err to target): {obs[2]: .4f}")
            print(f"       roll, pitch       : {obs[3]: .4f}, {obs[4]: .4f}")
            print(f"       p,q,r (ang rates) : {obs[5]: .4f}, {obs[6]: .4f}, {obs[7]: .4f}")

            target_z = obs[0] - obs[2]   # since dz = z - target_z
            print(f"\n  • Target altitude      : {target_z: .4f}")
            print(f"  • Current altitude     : {obs[0]: .4f}")
            print(f"  • Altitude error (dz)  : {obs[2]: .4f}")

            print("\n  • Info dict:")
            for k, v in info.items():
                print(f"       {k}: {v}")

            print("="*60 + "\n")


        print(f"[Eval] Episode {ep+1} | Step {eval_env.step_count}")

    eval_env.close()


if __name__ == "__main__":
    main()
