// src/bindings/python/bindings.cpp
#include <pybind11/pybind11.h>
#include "aeroforge_sim/sim_core.hpp"

namespace py = pybind11;

PYBIND11_MODULE(aeroforge_core, m) {
    m.doc() = "AeroForge core simulation bindings";

    py::class_<aeroforge::SimCore>(m, "SimCore")
        .def(py::init<>())
        .def("initialize", &aeroforge::SimCore::initialize)
        .def("reset", &aeroforge::SimCore::reset)
        .def("step", &aeroforge::SimCore::step);
}
