# pragma once

#include <string>
#include <vector>
#include <btBulletDynamicsCommon.h>

class ImuSensor{
    public:

    // Constrcutor 
    ImuSensor();

    // Destructor
    ~ImuSensor();

    // Initialize the IMU sensor
    bool initialize();

    // Read data from the IMU sensor
    bool readData();


    // IMU sensor parameters
    struct ImuConfig{
        // --- Noise characteristics ---
        double acceleration_noise_std;
        double gyro_noise_std;
        // --- Bias characteristics ---
        double accel_bias_rw_std_per_sec;
        double gyro_bias_rw_std_per_sec;
        // --- Scale clip factors ---
        double accel_scale_clip;
        double gyro_scale_clip;
    };

    // IMU reading structure
    struct ImuReading {
        btVector3 acceleration{}; // X, Y, Z acceleration
        btVector3 angular_velocity{}; // X, Y, Z angular velocity
        double timestamp; // Time of the reading
    };

    ImuReading ImuMeasure(std::vector<double> accel_body_, std::vector<double> gravity_world_,
               std::vector<double> q_world_body_, int dt );

    private:

};