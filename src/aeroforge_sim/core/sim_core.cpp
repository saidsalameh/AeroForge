// src/aeroforge_sim/core/sim_core.cpp
#include "aeroforge_sim/sim_core.hpp"
#include <iostream>

namespace aeroforge {

void SimCore::initialize() {
    std::cout << "[SimCore] initialize() called" << std::endl;
}

void SimCore::reset() {
    std::cout << "[SimCore] reset() called" << std::endl;
}

void SimCore::step() {
    std::cout << "[SimCore] step() called" << std::endl;
}

} // namespace aeroforge
