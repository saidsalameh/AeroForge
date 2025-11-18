import numpy as np

from aeroforge.core.simcore_loader import SimCore


class DroneNavEnv:
    """Drone navigation environment wrapping the C++ SimCore physics."""

    def __init__(self, max_steps: int = 500, target_position=None):
        # --- Core simulator ---
        self.sim = SimCore()
        self.sim.initialize()

        # --- Episode config ---
        self.max_steps = max_steps
        if target_position is None:
            target_position = [0.0, 0.0, 1.0]
        self.target_position = np.array(target_position, dtype=np.float64)

        # --- Internal state ---
        self.step_count = 0

        # --- (Optional) Gym-style spaces ---
        # We try to infer obs_dim from a sample observation
        try:
            import gymnasium as gym

            sample_obs = self.sim.get_observation()
            obs_dim = int(sample_obs.shape[0])

            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32,
            )

            # Placeholder 4D action space (e.g. thrust + body rates later)
            self.action_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(4,),
                dtype=np.float32,
            )
        except ImportError:
            # If gym isn't installed, just skip spaces definition
            self.observation_space = None
            self.action_space = None

    def reset(self):
        """Reset the environment to the initial state."""
        self.sim.reset()
        self.step_count = 0
        obs = self.get_observation()
        info = {}
        return obs, info

    def step(self, action):
        """Advance the environment by one step.

        For now, the action is ignored by the physics engine (no control yet).
        """
        # Placeholder: we ignore the action until SimCore supports it
        self.sim.step()

        self.step_count += 1

        obs = self.get_observation()
        reward = self.compute_reward(obs)
        done = self.check_done(obs)
        info = {
            "distance_to_target": float(self._distance_to_target(obs)),
        }
        return obs, reward, done, info

    def get_observation(self):
        """Return the current observation vector from the simulator.

        For Stage 2 this is just the raw 13D state from SimCore:
        [pos(3), quat(4), lin_vel(3), ang_vel(3)].
        """
        obs = self.sim.get_observation()
        return np.asarray(obs, dtype=np.float64)

    # ---- Helpers for reward and termination ----

    def _extract_position(self, obs: np.ndarray) -> np.ndarray:
        """Extract drone position [x, y, z] from the observation."""
        return obs[0:3]

    def _distance_to_target(self, obs: np.ndarray) -> float:
        pos = self._extract_position(obs)
        return float(np.linalg.norm(pos - self.target_position))

    def compute_reward(self, obs: np.ndarray) -> float:
        """Simple reward: negative distance to target."""
        d = self._distance_to_target(obs)
        return -d

    def check_done(self, obs: np.ndarray) -> bool:
        """Episode termination based on time limit and simple bounds."""
        if self.step_count >= self.max_steps:
            return True

        pos = self._extract_position(obs)

        # Simple safety bounds (can be tuned later)
        if np.linalg.norm(pos) > 50.0:   # too far from origin
            return True
        if pos[2] < -1.0:                # fell far below ground
            return True

        return False

    def close(self):
        """Optional cleanup hook."""
        # For now, nothing special to do.
        pass
