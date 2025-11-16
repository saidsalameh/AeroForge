#pragma once

// Bullet main header (brings in btVector3, btRigidBody, etc.)
#include <btBulletDynamicsCommon.h>

/// @brief SimCore
///
/// SimCore is the core C++ simulation engine for a single AeroForge world.
///
/// Responsibility:
///   - Own and manage the Bullet physics world (gravity, collision, solver).
///   - Create and reset the main simulation objects (ground, drone body).
///   - Advance the simulation in fixed time steps.
///   - Expose the current physical state (pose, velocities, etc.) as a
///     numeric observation to higher-level code (Python, RL, tools).
///
/// SimCore deliberately does NOT:
///   - Implement any RL logic, agents or training algorithms.
///   - Know about Gym, Stable-Baselines, or specific reward shaping.
///   - Handle rendering, GUI or VSLAM (these are separate layers).
///
/// Typical lifecycle:
///   1. Construct SimCore.
///   2. Call initialize() once to create the Bullet world and configure gravity.
///   3. For each episode:
///        a) reset() to place the drone and ground in their initial state.
///        b) Repeatedly:
///             - step(action) to advance the physics.
///             - getObservation(...) to fetch the current drone state.
///             - (later) computeReward(), isDone() for task logic.
///   4. On destruction, SimCore releases all Bullet resources.

namespace aeroforge {

class SimCore {
public:
    /// @brief Default constructor.
    SimCore() = default;

    /// @brief Destructor releases all Bullet resources owned by this instance.
    ~SimCore();

    /// @brief Initialize the Bullet physics world and global settings.
    void initialize();

    /// @brief Reset the simulation to the beginning of an episode.
    void reset();

    /// @brief Advance the simulation by one control step.
    void step();

    /// @brief Fill a flat observation vector with the current drone state.
    ///
    /// For Stage 1, the observation layout is:
    ///   [0..2]   position (x, y, z)
    ///   [3..6]   orientation quaternion (x, y, z, w)
    ///   [7..9]   linear velocity (vx, vy, vz)
    ///   [10..12] angular velocity (wx, wy, wz)
    ///
    /// So @p size must be at least 13.
    void getObservation(double* out, int size) const;

private:
    /// @brief Release all Bullet resources owned by this SimCore instance.
    void shutdown();

    // --- Bullet world components ---

    btDefaultCollisionConfiguration*      collision_config_   = nullptr;
    btCollisionDispatcher*                dispatcher_         = nullptr;
    btBroadphaseInterface*                broadphase_         = nullptr;
    btSequentialImpulseConstraintSolver*  solver_             = nullptr;
    btDiscreteDynamicsWorld*              dynamics_world_     = nullptr;

    // --- Main simulation bodies and shapes ---

    btRigidBody*      ground_body_ = nullptr;
    btRigidBody*      drone_body_  = nullptr;

    btCollisionShape* ground_shape_ = nullptr;
    btCollisionShape* drone_shape_  = nullptr;

    // --- Episode configuration / mission data ---

    btVector3    drone_initial_position_{0.0, 0.0, 1.0};
    btQuaternion drone_initial_orientation_{0.0, 0.0, 0.0, 1.0};

    // For Stage 1 the target is not used yet, but we keep it for future tasks.
    btVector3 target_position_{0.0, 0.0, 1.0};

    // --- Simulation timing and episode tracking ---

    int    step_count_            = 0;
    int    max_steps_per_episode_ = 1000;
    int    substeps_per_control_  = 10;
    double sim_dt_                = 0.01; // 10 ms per physics step
};

} // namespace aeroforge
