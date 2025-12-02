import numpy as np

from aeroforge.core.simcore_loader import SimCore


class DroneNavEnv:
    """
    Drone navigation environment wrapping the C++ SimCore physics.

    Stage 5 base: 3D navigation to a single target.

    Observation layout (15D):
        obs = [
            x, y, z,            # position in world frame
            vx, vy, vz,         # linear velocity in world frame
            roll, pitch, yaw,   # attitude (rad)
            p, q, r,            # body angular rates (rad/s)
            dx, dy, dz          # position error to target = pos - target_pos
        ]
    """

    def __init__(self, max_steps: int = 500, target_position=None):
        # --- Core simulator ---
        self.sim = SimCore()
        self.sim.initialize()

        # --- Episode config ---
        self.max_steps = max_steps

        if target_position is None:
            # Default target: 1 m above origin
            target_position = [0.0, 0.0, 1.0]

        self.target_position = np.array(target_position, dtype=np.float64)

        # --- Internal state ---
        self.step_count = 0
        self.last_distance = None  # for shaping
        self.last_dx = 0.0
        self.last_dy = 0.0
        self.last_dz = 0.0

        # --- Optional Gymnasium-style spaces ---
        try:
            import gymnasium as gym

            obs_dim = 15  # as described above
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
            self.observation_space = None
            self.action_space = None

    # -------------------------------------------------------------------------
    # Core RL API
    # -------------------------------------------------------------------------

    def reset(self):
        """Reset the environment to the initial state."""
        self.sim.reset()
        self.step_count = 0

        obs = self.get_observation()

        # Initialize shaping history
        self.last_dx = obs[12]
        self.last_dy = obs[13]
        self.last_dz = obs[14]
        self.last_distance = self._distance_to_target(obs)

        info = {
            "target_position": self.target_position.copy(),
        }
        return obs, info

    def step(self, action):
        """
        Advance the environment by one step.

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

        # 3) Apply to simulator and step physics
        self.sim.set_action(action)
        self.sim.step()
        self.step_count += 1

        # 4) Build RL outputs
        obs = self.get_observation()
        reward = self.compute_reward(obs)
        done = self.check_done(obs)

        d = self._distance_to_target(obs)
        info = {
            "distance_to_target": float(d),
            "target_position": self.target_position.copy(),
        }

        return obs, reward, done, info

    # -------------------------------------------------------------------------
    # Observation construction
    # -------------------------------------------------------------------------

    def to_Euler(self, quat):
        """
        Convert quaternion [x, y, z, w] to Euler angles (roll, pitch, yaw).

        Returns angles in radians.
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
        """
        Return the full 15D observation for 3D navigation.

        Layout:
            [x, y, z,
             vx, vy, vz,
             roll, pitch, yaw,
             p, q, r,
             dx, dy, dz]
        """
        # Raw layout from SimCore: [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        raw = np.asarray(self.sim.get_observation(), dtype=np.float64)
        pos = raw[0:3]
        quat = raw[3:7]
        lin_vel = raw[7:10]
        ang_vel = raw[10:13]

        x, y, z = pos
        vx, vy, vz = lin_vel
        p, q, r = ang_vel

        roll, pitch, yaw = self.to_Euler(quat)

        # Error to current target in world frame
        dx = x - self.target_position[0]
        dy = y - self.target_position[1]
        dz = z - self.target_position[2]

        obs = np.array(
            [
                x, y, z,
                vx, vy, vz,
                roll, pitch, yaw,
                p, q, r,
                dx, dy, dz,
            ],
            dtype=np.float64,
        )
        return obs

    # -------------------------------------------------------------------------
    # Helpers for reward and termination
    # -------------------------------------------------------------------------

    def _distance_to_target(self, obs: np.ndarray) -> float:
        """3D Euclidean distance from current position to target."""
        dx, dy, dz = obs[12:15]
        return float(np.sqrt(dx * dx + dy * dy + dz * dz))

    def compute_reward(self, obs: np.ndarray) -> float:
        """
        NASA/Airbus-inspired reward for stable 3D navigation to a single target.

        obs:
            [x, y, z,
             vx, vy, vz,
             roll, pitch, yaw,
             p, q, r,
             dx, dy, dz]
        """
        x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r, dx, dy, dz = obs

        # Core distances and magnitudes
        d = self._distance_to_target(obs)
        v = float(np.sqrt(vx * vx + vy * vy + vz * vz))
        att_mag = roll * roll + pitch * pitch  # yaw usually free for quadrotor

        # Initialize history if first call
        if self.last_distance is None:
            self.last_distance = d

        # --- Normalized potential terms (bounded in [-1, 0]) ---
        # Normalize distances and speeds to avoid huge penalties
        d_norm = min(d / 10.0, 1.0)    # assume workspace radius ~10 m
        v_norm = min(v / 5.0, 1.0)     # assume 5 m/s is "fast"
        att_norm = min(att_mag / 1.0, 1.0)  # ~1 rad^2 ≈ 57 deg^2 total tilt

        r_pos = -d_norm
        r_vel = -v_norm
        r_att = -att_norm

        # --- Radial progress shaping ---
        # Reward moving closer: Δd > 0 -> positive bonus
        delta_d = self.last_distance - d
        r_progress = 0.5 * delta_d

        # --- Smoothness penalty on body rates ---
        jerk_penalty = -0.01 * (p * p + q * q + r * r)

        # --- Terminal "arrival/park" bonus ---
        d_tol = 0.1    # 10 cm radius
        v_tol = 0.2    # m/s
        tilt_tol = 0.1 # rad (~6 deg)

        arrival_bonus = 0.0
        if (
            d < d_tol
            and v < v_tol
            and abs(roll) < tilt_tol
            and abs(pitch) < tilt_tol
        ):
            arrival_bonus = 5.0

        reward = (
            r_pos
            + r_vel
            + r_att
            + r_progress
            + jerk_penalty
            + arrival_bonus
        )

        # Update history
        self.last_distance = d
        self.last_dx = dx
        self.last_dy = dy
        self.last_dz = dz

        return float(reward)

    def check_done(self, obs: np.ndarray) -> bool:
        """
        Termination based on:
          - time limit
          - crash / out-of-bounds
          - excessive tilt
        """
        # 1) Time limit
        if self.step_count >= self.max_steps:
            return True

        x, y, z = obs[0:3]
        roll, pitch, yaw = obs[6:9]

        # 2) Crash: too low
        if z < 0.0:
            return True

        # 3) Out-of-bounds position
        if np.sqrt(x * x + y * y + z * z) > 50.0:
            return True

        # 4) Excessive tilt -> consider unstable
        if abs(roll) > np.deg2rad(80.0) or abs(pitch) > np.deg2rad(80.0):
            return True

        return False

    def close(self):
        """Optional cleanup hook."""
        # For now, nothing special to do.
        pass


# -------------------------------------------------------------------------
# Stage 6: Waypoint-based navigation environment
# -------------------------------------------------------------------------

class DroneWaypointEnv(DroneNavEnv):
    """
    Stage 6 environment: sequential waypoint navigation.

    The drone must visit waypoints in order:
        waypoints[0] -> waypoints[1] -> ... -> waypoints[N-1]

    At any time, the "target" in observation and reward is the
    CURRENT waypoint. When a waypoint is reached (within tolerance),
    we switch to the next one.
    """

    def __init__(self, waypoints=None, max_steps: int = 1000):
        # Default pattern: a square at z=1 m
        if waypoints is None:
            waypoints = np.array(
                [
                    [2.0, 2.0, 1.0],
                    [2.0, -2.0, 1.0],
                    [-2.0, -2.0, 1.0],
                    [-2.0, 2.0, 1.0],
                ],
                dtype=np.float64,
            )
        else:
            waypoints = np.asarray(waypoints, dtype=np.float64)
            assert waypoints.ndim == 2 and waypoints.shape[1] == 3, \
                "waypoints must be (N, 3)"

        self.waypoints = waypoints
        self.current_wp_idx = 0

        # Initialize base class with first waypoint as target
        super().__init__(max_steps=max_steps, target_position=self.waypoints[0])

        # waypoint-related hyperparameters
        self.wp_pos_tol = 0.1   # [m]
        self.wp_vel_tol = 0.5   # [m/s]
        self.wp_bonus = 5.0     # per waypoint
        self.mission_bonus = 10.0

    def _current_waypoint(self):
        return self.waypoints[self.current_wp_idx]

    def reset(self):
        """Reset mission: start at waypoint 0."""
        self.current_wp_idx = 0
        self.target_position = self.waypoints[self.current_wp_idx].copy()

        obs, info = super().reset()
        info.update(
            {
                "current_waypoint_index": int(self.current_wp_idx),
                "current_waypoint": self.target_position.copy(),
            }
        )
        return obs, info

    def get_observation(self):
        """
        Same 15D layout as DroneNavEnv, but dx, dy, dz are computed
        w.r.t the CURRENT waypoint.
        """
        raw = np.asarray(self.sim.get_observation(), dtype=np.float64)
        pos = raw[0:3]
        quat = raw[3:7]
        lin_vel = raw[7:10]
        ang_vel = raw[10:13]

        x, y, z = pos
        vx, vy, vz = lin_vel
        p, q, r = ang_vel

        roll, pitch, yaw = self.to_Euler(quat)

        current_wp = self._current_waypoint()
        dx = x - current_wp[0]
        dy = y - current_wp[1]
        dz = z - current_wp[2]

        obs = np.array(
            [
                x, y, z,
                vx, vy, vz,
                roll, pitch, yaw,
                p, q, r,
                dx, dy, dz,
            ],
            dtype=np.float64,
        )
        return obs

    def step(self, action):
        """
        Same as base, but with waypoint switching & mission completion.
        """
        obs, reward, done, info = super().step(action)

        # Recompute distance to current waypoint using updated obs
        d = self._distance_to_target(obs)
        vx, vy, vz = obs[3:6]
        speed = float(np.sqrt(vx * vx + vy * vy + vz * vz))

        wp_reached = False
        mission_complete = False

        # If close enough and slow enough: mark waypoint reached
        if d < self.wp_pos_tol and speed < self.wp_vel_tol:
            wp_reached = True
            reward += self.wp_bonus

            # If not last waypoints, move to next
            if self.current_wp_idx < len(self.waypoints) - 1:
                self.current_wp_idx += 1
                self.target_position = self.waypoints[self.current_wp_idx].copy()

                # Reset shaping distance w.r.t new target
                self.last_distance = self._distance_to_target(obs)
            else:
                # Last waypoint reached -> mission complete
                mission_complete = True
                reward += self.mission_bonus
                done = True

        info.update(
            {
                "current_waypoint_index": int(self.current_wp_idx),
                "current_waypoint": self._current_waypoint().copy(),
                "waypoint_reached": wp_reached,
                "mission_complete": mission_complete,
            }
        )

        return obs, reward, done, info
