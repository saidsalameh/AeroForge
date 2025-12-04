# pragma once

#include <string>
#include <vector>
#include <array>
#include "common/Types.h"
#include <btBulletDynamicsCommon.h>
#include <ctime>
#include <iostream>
#include <random>

class ImuSensor{
    public:

    // -----------------------------------------
    // Method to set IMU configuration
    // -----------------------------------------

    // Constrcutor 
    ImuSensor();

    // Destructor
    ~ImuSensor();

    // Initialize the IMU sensor
    bool initialize(const ImuConfig& config, unsigned int seed);


    // IMU reading structure
    struct ImuReading {
        array3 acceleration{}; // X, Y, Z acceleration body-frame specific force [m/s²]
        array3 angular_velocity{}; // X, Y, Z angular velocity body-frame angular rate [rad/s]
        double timestamp; // Time of the reading in seconds
    };

    //-----------------------------------------
    // IMU configuration stucture
    // -----------------------------------------
    struct AxisErrorConfig {
        
        double mis_xy, mis_xz;
        double mis_yx, mis_yz;
        double mis_zx, mis_zy;

        std::array<double, 3> bias_constant;            // accel: m/s² | gyro: rad/s
        std::array<double, 3> bias_rw_std_per_s;        // accel: m/s²/√s | gyro: rad/s/√s
        std::array<double, 3> scale_factor_error;       // accel: m/s² | gyro: rad/s
        std::array<double, 3> noise_std;                // accel: m/s²/√Hz | gyro: rad/s/√Hz
        std::array<double, 3> quantization;             // accel: m/s² | gyro: rad/s
        std::array<double, 3> rtn_amplitude;            // accel: m/s² | gyro: rad/s
        std::array<double, 3> clipping;                 // accel: m/s² | gyro: rad/s
        std::array<double, 3> rtn_mean_dwell_time_s;    // seconds
    };

    struct AxisRuntimeState{

        std::array<double, 3> bias_rw_current;          // Current random walk bias value
        std::array<double, 3> rtn_state_current;        // Current run-to-run noise value
        std::mt19937 rng;
        std::normal_distribution<double> normal01;      // normal01 should always be N(0,1)
        std::uniform_real_distribution<double> uniform01;// uniform01 should always be U(0,1)
    };

    struct ImuConfig {
        std::string sensor_name;
        double frequency;                               // IMU update frequency [Hz]
        // Error configuration
        AxisErrorConfig accelerometer;
        AxisErrorConfig gyroscope;


    };

    

    ImuReading measure(const btVector3& true_accel_body,
                   const btVector3& gravity_world,
                   const btQuaternion& orientation_world_to_body,
                   const btVector3& true_gyro_body,
                   double current_time_s,
                   double dt);

    private:

    ImuConfig m_config;
    
    AxisRuntimeState accelerometer_state;
    AxisRuntimeState gyroscope_state;
    unsigned int m_base_seed;                           // Base seed for random number generation
    
};

