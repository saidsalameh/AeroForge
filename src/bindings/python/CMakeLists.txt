# src/bindings/python/CMakeLists.txt

pybind11_add_module(aeroforge_core
    bindings.cpp
)

target_link_libraries(aeroforge_core
    PRIVATE
        aeroforge_sim
)
