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
# Expected path of this file:
#   python/aeroforge/scripts/train_hover_ppo.py
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
    Adapter to make DroneNavEnv compatible with Gymnasium / Stable-Baselines3.

    DroneNavEnv API:
        reset() -> (obs, info)
        step(action) -> (obs, reward, done, info)

    Gymnasium API:
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

        # Gymnasium separates termination reasons:
        terminated = bool(done)   # episode ended due to task/failure
        truncated = False         # we treat time-limit as normal termination

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
# Training configuration (Stage 5: 3D navigation)
# ---------------------------------------------------------------------------

N_ENVS = 4
TOTAL_TIMESTEPS = 100_000

LEARNING_RATE = 3e-4
N_STEPS = 1024
BATCH_SIZE = 256
GAMMA = 0.99

# Use Stage-5 specific directories to avoid mixing with hover-only runs
LOG_DIR = ROOT / "logs" / "nav3d_ppo"
MODEL_DIR = ROOT / "models" / "nav3d_ppo"
MODEL_PATH = MODEL_DIR / "ppo_drone_nav3d"

# Prepare log file
LOG_FILE = pathlib.Path(__file__).resolve().parents[3] / "tests" / "logs" / "train_output.txt"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# Duplicate stdout → file + terminal
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


def main():
    # Create output directories if they don't exist
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)



    # Redirect output
    sys.stdout = Tee(sys.stdout, open(LOG_FILE, "w"))
    sys.stderr = Tee(sys.stderr, open(LOG_FILE, "a"))

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

    # Observation layout in Stage 5:
    # obs = [
    #   0: x,  1: y,  2: z,
    #   3: vx, 4: vy, 5: vz,
    #   6: roll, 7: pitch, 8: yaw,
    #   9: p, 10: q, 11: r,
    #   12: dx, 13: dy, 14: dz
    # ]

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

            x, y, z = obs[0], obs[1], obs[2]
            vx, vy, vz = obs[3], obs[4], obs[5]
            roll, pitch, yaw = obs[6], obs[7], obs[8]
            p, q, r = obs[9], obs[10], obs[11]
            dx, dy, dz = obs[12], obs[13], obs[14]

            target_x = x - dx
            target_y = y - dy
            target_z = z - dz

            print("\n" + "============================================================")
            print(f"[Eval] Episode {ep+1} | Step {eval_env.step_count}")
            print("-------------------------------------------------------------")

            print(f"  • Reward this step         : {reward: .4f}")
            print(f"  • Cumulative reward        : {ep_reward: .4f}")

            print("\n  • Position & velocity:")
            print(f"       pos (x, y, z)         : {x: .4f}, {y: .4f}, {z: .4f}")
            print(f"       vel (vx, vy, vz)      : {vx: .4f}, {vy: .4f}, {vz: .4f}")

            print("\n  • Attitude & rates:")
            print(f"       roll, pitch, yaw      : {roll: .4f}, {pitch: .4f}, {yaw: .4f}")
            print(f"       p, q, r (ang rates)   : {p: .4f}, {q: .4f}, {r: .4f}")

            print("\n  • Target tracking:")
            print(f"       target (x, y, z)      : {target_x: .4f}, {target_y: .4f}, {target_z: .4f}")
            print(f"       error  (dx, dy, dz)   : {dx: .4f}, {dy: .4f}, {dz: .4f}")

            print("\n  • Info dict:")
            for k, v in info.items():
                print(f"       {k}: {v}")

            print("="*60 + "\n")

        print(f"[Eval] Episode {ep+1}/{n_eval_episodes} | Final step {eval_env.step_count}")
        print(f"[Eval] Episode {ep+1} total reward = {ep_reward: .4f}")

    eval_env.close()


if __name__ == "__main__":
    main()
