import numpy as np

from aeroforge.core.simcore_loader import SimCore


class DroneNavEnv:
    """Drone navigation environment wrapping the C++ SimCore physics.

    Stage 4: Hover task at target altitude (default z = 10 m).

    Observation layout (8D):
        obs = [ z, vz, dz, roll, pitch, p, q, r ]

    where:
        z      = altitude (m)
        vz     = vertical velocity (m/s)
        dz     = z - z_target (altitude error, m)
        roll   = roll angle (rad)
        pitch  = pitch angle (rad)
        p, q, r= body angular rates (rad/s)
    """

    def __init__(self, max_steps: int = 500, target_position=None):
        # --- Core simulator ---
        self.sim = SimCore()
        self.sim.initialize()

        # --- Episode config ---
        self.max_steps = max_steps
        if target_position is None:
            # Hover target at 1 m above origin
            target_position = [0.0, 0.0, 1.0]
        self.target_position = np.array(target_position, dtype=np.float64)
        
        self.last_distance = None # To compue reward difference
        # --- Internal state ---
        self.step_count = 0

        # --- Optional Gymnasium-style spaces ---
        try:
            import gymnasium as gym

            # Reduced obs is 8D: [z, vz, dz, roll, pitch, p, q, r]
            obs_dim = 8

            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32,
            )

            # 4D normalized action: [thrust, roll_rate, pitch_rate, yaw_rate] in [-1, 1]
            self.action_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(4,),
                dtype=np.float32,
            )
        except ImportError:
            # If gymnasium isn't installed, just skip spaces definition
            self.observation_space = None
            self.action_space = None

    # -------------------------------------------------------------------------
    # Core RL API
    # -------------------------------------------------------------------------

    def reset(self):
        """Reset the environment to the initial state."""
        self.sim.reset()
        # Inform SimCore of the target (for future use if needed)
        # self.sim.set_target_position(self.target_position.tolist())

        # For Stage 4, we assume SimCore.reset() places the drone
        # in a reasonable initial state near the target.
        self.step_count = 0

        obs = self.get_observation()
        self.last_distance = self._distance_to_target(obs)
        info = {}
        return obs, info

    def step(self, action):
        """Advance the environment by one step using the given action.

        Parameters
        ----------
        action : array-like of shape (4,)
            Normalized control inputs in [-1, 1]:
            [thrust_cmd, roll_rate_cmd, pitch_rate_cmd, yaw_rate_cmd]
        """
        # 1) Convert to numpy and validate shape
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (4,):
            raise ValueError(f"Action must be a 4D vector, got shape {action.shape}")

        # 2) Clip to normalized range [-1, 1]
        action = np.clip(action, -1.0, 1.0)

        # 3) Pass to simulator and advance physics
        self.sim.set_action(action)
        self.sim.step()
        self.step_count += 1

        # 4) Build RL outputs
        obs = self.get_observation()
        reward = self.compute_reward(obs)
        done = self.check_done(obs)

        # 5) Additional info (optional)
        # Computing distance to target for info
        self.last_distance = self._distance_to_target(obs)


        info = {
            "distance_to_target": float(self._distance_to_target(obs)),  # 1D altitude error
        }
        return obs, reward, done, info

    # -------------------------------------------------------------------------
    # Observation construction
    # -------------------------------------------------------------------------

    def to_Euler(self, quat):
        """Convert quaternion [x, y, z, w] to Euler angles (roll, pitch, yaw).

        Uses standard aerospace convention:
            roll  = rotation about X
            pitch = rotation about Y
            yaw   = rotation about Z
        """
        quat = np.asarray(quat, dtype=np.float64)
        x, y, z, w = quat  # Bullet returns (x, y, z, w)

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch_y = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)

        return np.array([roll_x, pitch_y, yaw_z], dtype=np.float64)

    def get_observation(self):
        """Return reduced observation for the hover task.

        Layout:
            [ z, vz, dz, roll, pitch, p, q, r ]
        """
        # Raw layout from SimCore: [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        raw = np.asarray(self.sim.get_observation(), dtype=np.float64)

        pos = raw[0:3]
        quat = raw[3:7]
        lin_vel = raw[7:10]
        ang_vel = raw[10:13]

        z = float(pos[2])
        vz = float(lin_vel[2])

        roll_x, pitch_y, yaw_z = self.to_Euler(quat)

        # Altitude error: current altitude minus target altitude
        target_z = float(self.target_position[2])
        dz = z - target_z

        reduced_obs = np.array(
            [z, vz, dz, roll_x, pitch_y, ang_vel[0], ang_vel[1], ang_vel[2]],
            dtype=np.float64,
        )
        return reduced_obs

    # -------------------------------------------------------------------------
    # Helpers for reward and termination
    # -------------------------------------------------------------------------

    def _distance_to_target(self, obs: np.ndarray) -> float:
        """1D distance in altitude between current z and target z."""
        # obs layout: [z, vz, dz, roll, pitch, p, q, r]
        dz = float(obs[2])  # z - z_target
        return abs(dz)

    def compute_reward(self, obs: np.ndarray) -> float:
        """Reward for hover at target altitude.

        obs layout: [z, vz, dz, roll, pitch, p, q, r]
        """
        z, vz, dz, roll, pitch, p, q, r = obs
        d = self._distance_to_target(obs)
        r_progress = 0.0
        
        # --- Reward weights (can be tuned or moved to __init__) ---
        w_z = 0.1     # altitude error weight
        w_v = 0.1     # vertical speed weight
        w_att = 0.1   # tilt weight
        w_rate = 0.01 # angular rate weight

        # Altitude error penalty (quadratic)
        r_alt = -w_z * (dz ** 2)

        # Penalize vertical speed (absolute)
        r_vz = -w_v * abs(vz)

        # Penalize tilt (roll & pitch)
        r_tilt = -w_att * (roll ** 2 + pitch ** 2)

        # Penalize angular rates
        r_rate = -w_rate * (p ** 2 + q ** 2 + r ** 2)

        
        if self.last_distance is not None:
            # Reward for reducing distance to target
            r_progress = 0.1 * (self.last_distance - d)
        self.last_distance = d

    
        # Total reward
        reward = r_alt + r_vz + r_tilt + r_rate + r_progress

        return float(reward)

    def check_done(self, obs: np.ndarray) -> bool:
        """Episode termination based on time limit and altitude bounds."""
        # 1) Time limit
        if self.step_count >= self.max_steps:
            return True

        # obs layout: [z, vz, dz, roll, pitch, p, q, r]
        z = float(obs[0])

        # 2) Crash: too low (below ground / crash threshold)
        z_crash = 0.1  # meters above ground considered "crashed"
        if z < z_crash:
            return True

        # 3) Optional: too high / escaped
        z_max = 50.0  # upper safety bound
        if z > z_max:
            return True

        return False

    def close(self):
        """Optional cleanup hook."""
        # For now, nothing special to do.
        pass
# End of drone_nav_env.py