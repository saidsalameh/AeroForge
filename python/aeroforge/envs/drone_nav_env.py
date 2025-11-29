import numpy as np
from math import sqrt

from aeroforge.core.simcore_loader import SimCore


class DroneNavEnv:
    """Drone navigation environment wrapping the C++ SimCore physics.

    Stage 5: 3D navigation to a fixed target position.

    Observation layout (15D):

        obs = [
            x, y, z,                 # 0–2   position (world)
            vx, vy, vz,              # 3–5   linear velocity
            roll, pitch, yaw,        # 6–8   attitude (rad)
            p, q, r,                 # 9–11  angular rates (rad/s)
            dx, dy, dz               # 12–14 position error to target
        ]

    where:
        [dx, dy, dz] = [x - x_target, y - y_target, z - z_target].
    """

    def __init__(self, max_steps: int = 500, target_position=None):
        # --- Core simulator ---
        self.sim = SimCore()
        self.sim.initialize()

        # --- Episode config ---
        self.max_steps = max_steps

        # Previous distance components
        self.last_dx = 0.0
        self.last_dy = 0.0
        self.last_dz = 0.0

        if target_position is None:
            # 3D target: x,y,z
            target_position = [12.0, 20.0, 5.0]
        self.target_position = np.array(target_position, dtype=np.float64)

        # Distance to target at previous step (for progress reward)
        self.last_distance = None

        # --- Internal state ---
        self.step_count = 0

        # --- Optional Gymnasium-style spaces ---
        try:
            import gymnasium as gym

            # 15D observation: [x,y,z, vx,vy,vz, roll,pitch,yaw, p,q,r, dx,dy,dz]
            obs_dim = 15

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

        info = {
            "distance_to_target": float(self._distance_to_target(obs)),
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
        """Return 15D observation for 3D navigation.

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

        # Positions
        x = float(pos[0])
        y = float(pos[1])
        z = float(pos[2])

        # Linear velocity
        vx = float(lin_vel[0])
        vy = float(lin_vel[1])
        vz = float(lin_vel[2])

        # Attitude
        roll_x, pitch_y, yaw_z = self.to_Euler(quat)

        # Angular rates
        p = float(ang_vel[0])
        q = float(ang_vel[1])
        r = float(ang_vel[2])

        # Position error to target
        dx = x - float(self.target_position[0])
        dy = y - float(self.target_position[1])
        dz = z - float(self.target_position[2])

        obs = np.array(
            [x, y, z,
             vx, vy, vz,
             roll_x, pitch_y, yaw_z,
             p, q, r,
             dx, dy, dz],
            dtype=np.float64,
        )
        return obs

    # -------------------------------------------------------------------------
    # Helpers for reward and termination
    # -------------------------------------------------------------------------

    def _distance_to_target(self, obs: np.ndarray) -> float:
        """3D Euclidean distance between current position and target."""
        # obs[12:15] are [dx, dy, dz]
        dx, dy, dz = obs[12:15]
        return float(sqrt(dx * dx + dy * dy + dz * dz))

    def compute_reward(self, obs: np.ndarray) -> float:
        """Reward for 3D navigation to target.

        obs layout:
            [x, y, z,
            vx, vy, vz,
            roll, pitch, yaw,
            p, q, r,
            dx, dy, dz]

        where:
            dx, dy, dz = position error components (pos - target_pos)
        """
        x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r, dx, dy, dz = obs

        # 3D distance to target
        d = float(np.sqrt(dx * dx + dy * dy + dz * dz))

        # ---------------------------------------------------------------------
        # Initialize history (first step of episode)
        # ---------------------------------------------------------------------
        if getattr(self, "last_distance", None) is None:
            self.last_distance = d
            self.last_dx = dx
            self.last_dy = dy
            self.last_dz = dz

        # ---------------------------------------------------------------------
        # Reward weights (tuneable hyperparameters)
        # Inspired by typical aerospace RL shaping:
        # - strong quadratic position penalty
        # - moderate velocity and attitude penalties
        # - small angular-rate penalty
        # - shaped progress term + terminal bonus
        # ---------------------------------------------------------------------
        w_pos = 1.0      # position error penalty (quadratic distance)
        w_vel = 0.2      # linear velocity magnitude
        w_tilt = 0.2     # roll/pitch attitude
        w_rate = 0.02    # body rates
        w_progress = 1.0 # radial progress shaping
        w_axis = 0.3     # per-axis progress shaping

        arrival_bonus_value = 10.0  # terminal bonus when "parked" at target

        # ---------------------------------------------------------------------
        # Base penalties
        # ---------------------------------------------------------------------
        # 1) Quadratic distance to target (potential function)
        r_pos = -w_pos * (d ** 2)

        # 2) Penalize speed (we prefer slow, controlled motion near target)
        speed_sq = vx * vx + vy * vy + vz * vz
        r_vel = -w_vel * speed_sq

        # 3) Penalize attitude deviations (keep drone "level")
        r_tilt = -w_tilt * (roll * roll + pitch * pitch)

        # 4) Penalize angular rates (avoid aggressive spinning)
        r_rate = -w_rate * (p * p + q * q + r * r)

        # ---------------------------------------------------------------------
        # Radial distance progress (NASA-style potential shaping)
        # F(s) = -k * d^2  -> reward_shaping ≈ F(s') - F(s)
        # Here we use a simple linear term on d for stability.
        # ---------------------------------------------------------------------
        r_progress_radial = 0.0
        if self.last_distance is not None:
            delta_d = self.last_distance - d  # >0 if we moved closer
            r_progress_radial = w_progress * delta_d

        # ---------------------------------------------------------------------
        # Per-axis progress (Airbus-style: directional guidance)
        # Reward only improvements in |dx|, |dy|, |dz| (moving toward target).
        # ---------------------------------------------------------------------
        r_progress_axis = 0.0
        last_abs_dx = abs(self.last_dx)
        last_abs_dy = abs(self.last_dy)
        last_abs_dz = abs(self.last_dz)

        abs_dx = abs(dx)
        abs_dy = abs(dy)
        abs_dz = abs(dz)

        delta_abs_dx = last_abs_dx - abs_dx
        delta_abs_dy = last_abs_dy - abs_dy
        delta_abs_dz = last_abs_dz - abs_dz

        # Only reward progress (positive improvement); ignore regress
        axis_progress_sum = max(delta_abs_dx, 0.0) \
                        + max(delta_abs_dy, 0.0) \
                        + max(delta_abs_dz, 0.0)

        r_progress_axis = w_axis * axis_progress_sum

        # ---------------------------------------------------------------------
        # Arrival / "parked" bonus near target
        # ---------------------------------------------------------------------
        d_tol = 0.1   # 10 cm radius around target
        v_tol = 0.2   # m/s
        tilt_tol = 0.2  # rad (~11 deg)

        speed_mag = float(np.sqrt(speed_sq))

        arrival_bonus = 0.0
        if (
            d < d_tol
            and speed_mag < v_tol
            and abs(roll) < tilt_tol
            and abs(pitch) < tilt_tol
        ):
            arrival_bonus = arrival_bonus_value

        # ---------------------------------------------------------------------
        # Total reward
        # ---------------------------------------------------------------------
        reward = (
            r_pos
            + r_vel
            + r_tilt
            + r_rate
            + r_progress_radial
            + r_progress_axis
            + arrival_bonus
        )

        # ---------------------------------------------------------------------
        # Update history for next step
        # ---------------------------------------------------------------------
        self.last_distance = d
        self.last_dx = dx
        self.last_dy = dy
        self.last_dz = dz

        return float(reward)


    def check_done(self, obs: np.ndarray) -> bool:
        """Episode termination based on time limit and safety bounds."""
        # 1) Time limit
        if self.step_count >= self.max_steps:
            return True

        # obs layout:
        # [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r, dx, dy, dz]
        x, y, z = float(obs[0]), float(obs[1]), float(obs[2])
        d = self._distance_to_target(obs)

        # 2) Success: close enough to target
        d_tol = 0.1
        if d < d_tol:
            return True

        # 3) Crash: too low (below ground / crash threshold)
        z_crash = 0.1  # meters above ground considered "crashed"
        if z < z_crash:
            return True

        # 4) Out-of-bounds in position
        max_radius = 50.0
        if sqrt(x * x + y * y + z * z) > max_radius:
            return True

        # 5) Too high
        z_max = 50.0
        if z > z_max:
            return True

        return False

    def close(self):
        """Optional cleanup hook."""
        # For now, nothing special to do.
        pass
