pybind11_add_module(aeroforge_core
    bindings.cpp
)

target_link_libraries(aeroforge_core
    PRIVATE
        aeroforge_sim
)

target_include_directories(aeroforge_core
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/external/bullet/src
)
