#include "aeroforge_sim/sim_core.hpp"
#include <iostream>
#include <algorithm>
#include <iostream>

namespace aeroforge {

SimCore::~SimCore() {
    shutdown();
}

void SimCore::setAction(const double* action, int size )
{
    // Validate drone body and input
    if(!drone_body_){
        std::cerr << "[Simcore] setAction: drone body not initialized \n";
        return;
    }

    if(!action || size < 4){
        std::cerr << "[Simcore] setAction: invalid action input \n";
        return;
    }

    // Map normalized action to physical units
    // Collective thrust: map from [-1, 1] to [0, max_thrust]
    auto clamped_thrust = [](double u){
        return std::max(0.0, std::min(1.0, (u + 1.0) / 2.0));
    };

    const double u_thrust   = clamped_thrust(action[0]);
    const double u_p        = clamped_thrust(action[1]);
    const double u_q        = clamped_thrust(action[2]);
    const double u_r        = clamped_thrust(action[3]);    

    // Map normalized thrust to [thrust_min_, thrust_max_]
    // u_thrust = -1 -> thrust_min_
    // u_thrust = +1 -> thrust_max_
    const double half_range = 0.5 * (max_thrust_ - min_thrust_);
    const double mid        = 0.5 * (max_thrust_ + min_thrust_);
    thrust_cmd_             = mid + half_range * u_thrust;

    // Map normalized body rates to [-max_rate, +max_rate] in rad/s
    const double p_cmd = u_p * max_roll_rate_;
    const double q_cmd = u_q * max_pitch_rate_;
    const double r_cmd = u_r * max_yaw_rate_;

    rate_cmd_.setValue(p_cmd, q_cmd, r_cmd);

}

void SimCore::initialize() {
    // Clean any previous world if re-initialized
    shutdown();

    // 1) Create Bullet world components
    collision_config_ = new btDefaultCollisionConfiguration();
    dispatcher_       = new btCollisionDispatcher(collision_config_);
    broadphase_       = new btDbvtBroadphase();
    solver_           = new btSequentialImpulseConstraintSolver();

    // 2) Create the dynamics world and set gravity
    dynamics_world_ = new btDiscreteDynamicsWorld(
        dispatcher_, broadphase_, solver_, collision_config_
    );
    dynamics_world_->setGravity(btVector3(0, 0, -9.81));

    // Reset episode counters
    step_count_ = 0;

    std::cout << "[SimCore] initialize(): Bullet world created\n";
}

void SimCore::shutdown() {
    if (dynamics_world_) {
        // Remove and delete all rigid bodies in the world
        for (int i = dynamics_world_->getNumCollisionObjects() - 1; i >= 0; --i) {
            btCollisionObject* obj = dynamics_world_->getCollisionObjectArray()[i];
            btRigidBody* body      = btRigidBody::upcast(obj);

            if (body && body->getMotionState()) {
                delete body->getMotionState();
            }

            dynamics_world_->removeCollisionObject(obj);
            delete obj;
        }

        delete dynamics_world_;
        dynamics_world_ = nullptr;
    }

    // Delete shapes
    delete ground_shape_;
    ground_shape_ = nullptr;

    delete drone_shape_;
    drone_shape_ = nullptr;

    // Delete world components
    delete solver_;
    solver_ = nullptr;

    delete broadphase_;
    broadphase_ = nullptr;

    delete dispatcher_;
    dispatcher_ = nullptr;

    delete collision_config_;
    collision_config_ = nullptr;

    std::cout << "[SimCore] shutdown(): Bullet world destroyed\n";
}

void SimCore::reset() {
    if (!dynamics_world_) {
        std::cerr << "[SimCore] reset(): dynamics_world_ is null, did you call initialize()?\n";
        return;
    }

    // Remove existing bodies (but keep world components)
    // --------------------------------------------------
    for (int i = dynamics_world_->getNumCollisionObjects() - 1; i >= 0; --i) {
        btCollisionObject* obj = dynamics_world_->getCollisionObjectArray()[i];
        btRigidBody* body      = btRigidBody::upcast(obj);

        if (body && body->getMotionState()) {
            delete body->getMotionState();
        }

        dynamics_world_->removeCollisionObject(obj);
        delete obj;
    }

    // Delete old shapes if any
    delete ground_shape_;
    ground_shape_ = nullptr;

    delete drone_shape_;
    drone_shape_ = nullptr;

    // --- Create ground plane ---
    ground_shape_ = new btStaticPlaneShape(btVector3(0, 0, 1), 0); // z = 0 plane

    btTransform ground_transform;
    ground_transform.setIdentity();
    ground_transform.setOrigin(btVector3(0, 0, 0));

    btDefaultMotionState* ground_motion_state =
        new btDefaultMotionState(ground_transform);

    btRigidBody::btRigidBodyConstructionInfo ground_ci(
        0.0,                        // mass
        ground_motion_state,        // motion state
        ground_shape_,              // collision shape
        btVector3(0, 0, 0)          // local inertia (zero for static)
    );

    ground_body_ = new btRigidBody(ground_ci);
    dynamics_world_->addRigidBody(ground_body_);

    // --- Create drone body (simple box) ---
    drone_shape_ = new btBoxShape(btVector3(0.1, 0.1, 0.05)); // half extents

    btTransform drone_transform;
    drone_transform.setIdentity();
    drone_transform.setOrigin(drone_initial_position_);
    drone_transform.setRotation(drone_initial_orientation_);

    btDefaultMotionState* drone_motion_state =
        new btDefaultMotionState(drone_transform);

    btScalar  drone_mass   = 1.0;
    btVector3 drone_inertia(0, 0, 0);
    drone_shape_->calculateLocalInertia(drone_mass, drone_inertia);

    btRigidBody::btRigidBodyConstructionInfo drone_ci(
        drone_mass,
        drone_motion_state,
        drone_shape_,
        drone_inertia
    );

    drone_body_ = new btRigidBody(drone_ci);
    dynamics_world_->addRigidBody(drone_body_);

    // Reset step count
    step_count_ = 0;

    std::cout << "[SimCore] reset(): ground + drone created\n";
}

void SimCore::step() {
    if (!dynamics_world_) {
        std::cerr << "[SimCore] step(): dynamics_world_ is null, did you call initialize()?\n";
        return;
    }

    // If for some reason the drone body is missing, still step the world
    if (!drone_body_) {
        std::cerr << "[SimCore] step(): drone body not initialized\n";
        dynamics_world_->stepSimulation(sim_dt_, substeps_per_control_);
        ++step_count_;
        return;
    }

    // --- 1) Thrust force in world frame ---

    btTransform transform;
    drone_body_->getMotionState()->getWorldTransform(transform);

    // Unit z-axis in body frame
    const btVector3 up_body(0, 0, 1);
    // Corresponding direction in world frame
    const btVector3 up_world = transform.getBasis() * up_body;

    // Thrust vector in world frame (Newtons)
    const btVector3 thrust_force = up_world * thrust_cmd_;

    // Apply central thrust force (Bullet will add gravity itself)
    drone_body_->applyCentralForce(thrust_force);

    // --- 2) Simple P rate controller for attitude ---

    // Angular velocity from Bullet (approx as body rates)
    const btVector3 omega = drone_body_->getAngularVelocity();
    const btScalar p = omega.getX();
    const btScalar q = omega.getY();
    const btScalar r = omega.getZ();

    const btScalar e_p = rate_cmd_.getX() - p;
    const btScalar e_q = rate_cmd_.getY() - q;
    const btScalar e_r = rate_cmd_.getZ() - r;

    const btScalar tau_x = kp_p_ * e_p;
    const btScalar tau_y = kp_q_ * e_q;
    const btScalar tau_z = kp_r_ * e_r;

    btVector3 torque_body(tau_x, tau_y, tau_z);

    // Optional: clamp torque magnitude to avoid insane spins
    const btScalar max_torque = 1.0; // N·m, rough bound
    if (torque_body.length() > max_torque) {
        torque_body = torque_body.normalized() * max_torque;
    }

    drone_body_->applyTorque(torque_body);

    // --- 3) Advance physics ---
    dynamics_world_->stepSimulation(sim_dt_, substeps_per_control_);
    ++step_count_;

}

void SimCore::getObservation(double* out, int size) const {
    if (!drone_body_ || !drone_body_->getMotionState()) {
        std::cerr << "[SimCore] getObservation(): drone body not initialized\n";
        return;
    }

    if (size < 13) {
        std::cerr << "[SimCore] getObservation(): output buffer too small (size < 13)\n";
        return;
    }

    btTransform transform;
    drone_body_->getMotionState()->getWorldTransform(transform);

    btVector3    position        = transform.getOrigin();
    btQuaternion orientation     = transform.getRotation();
    btVector3    linear_velocity = drone_body_->getLinearVelocity();
    btVector3    angular_velocity= drone_body_->getAngularVelocity();

    // Position
    out[0] = position.getX();
    out[1] = position.getY();
    out[2] = position.getZ();

    // Orientation (quaternion x,y,z,w)
    out[3] = orientation.getX();
    out[4] = orientation.getY();
    out[5] = orientation.getZ();
    out[6] = orientation.getW();

    // Linear velocity
    out[7]  = linear_velocity.getX();
    out[8]  = linear_velocity.getY();
    out[9]  = linear_velocity.getZ();

    // Angular velocity
    out[10] = angular_velocity.getX();
    out[11] = angular_velocity.getY();
    out[12] = angular_velocity.getZ();
}

} // namespace aeroforge
