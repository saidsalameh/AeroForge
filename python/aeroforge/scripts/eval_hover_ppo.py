import sys
import pathlib
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

# ---------- Path setup ----------
HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[3]  # .../AeroForge

PYTHON_PACKAGE_DIR = ROOT / "python"
BINDINGS_DIR = ROOT / "build" / "src" / "bindings" / "python"

sys.path.insert(0, str(PYTHON_PACKAGE_DIR))
sys.path.insert(0, str(BINDINGS_DIR))

from aeroforge.envs.drone_nav_env import DroneNavEnv


class DroneNavGymEnv(gym.Env):
    """Same wrapper as in train_hover_ppo.py, but minimal for eval."""

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 500):
        super().__init__()
        self._env = DroneNavEnv(max_steps=max_steps)
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
        truncated = False
        return obs, reward, terminated, truncated, info

    def close(self):
        self._env.close()
        super().close()

    @property
    def step_count(self):
        return self._env.step_count


def main():
    MODEL_PATH = ROOT / "models" / "hover_ppo" / "ppo_drone_hover.zip"

    # 1) Create env
    env = DroneNavGymEnv(max_steps=500)

    # 2) Load trained model
    model = PPO.load(str(MODEL_PATH))

    # 3) Run evaluation episodes
    n_eval_episodes = 5
    for ep in range(n_eval_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward

            # Optional: compact logging
            z, vz, dz = obs[2], obs[5], obs[14]  # adapt indices if needed
            print(f"[Eval] ep={ep+1:02d} step={env.step_count:03d} "
                  f"reward={reward: .3f} z={z: .3f} dz={dz: .3f}")

        print(f"[Eval] Episode {ep+1}/{n_eval_episodes}, total reward = {ep_reward:.3f}")

    env.close()


if __name__ == "__main__":
    main()
