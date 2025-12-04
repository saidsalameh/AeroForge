#include "include/sensors/ImuSensor.h"

ImuSensor::ImuSensor(){
    // Constructor implementation (if needed)
};

ImuSensor::~ImuSensor(){
    // Destructor implementation (if needed)
};

bool ImuSensor::initialize(const ImuConfig&, unsigned int seed){
    // Initialization code for the IMU sensor
    return true; // Return true if initialization is successful
}

