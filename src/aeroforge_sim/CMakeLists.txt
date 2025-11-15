# src/aeroforge_sim/CMakeLists.txt

add_library(aeroforge_sim STATIC
    core/sim_core.cpp
)

target_include_directories(aeroforge_sim
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)
