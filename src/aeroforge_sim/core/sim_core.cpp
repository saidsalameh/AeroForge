#include "aeroforge_sim/sim_core.hpp"
#include <iostream>

namespace aeroforge {

SimCore::~SimCore() {
    shutdown();
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
