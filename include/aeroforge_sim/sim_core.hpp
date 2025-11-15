// include/aeroforge_sim/sim_core.hpp
#pragma once

namespace aeroforge {

class SimCore {
public:
    SimCore() = default;
    ~SimCore() = default;

    void initialize();
    void reset();
    void step();
};

} // namespace aeroforge
