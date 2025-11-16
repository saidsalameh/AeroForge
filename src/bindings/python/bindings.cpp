// src/bindings/python/bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <array>

#include "aeroforge_sim/sim_core.hpp"

namespace py = pybind11;

PYBIND11_MODULE(aeroforge_core, m) {
    m.doc() = "AeroForge core simulation bindings";

    py::class_<aeroforge::SimCore>(m, "SimCore")
        .def(py::init<>())
        .def("initialize", &aeroforge::SimCore::initialize)
        .def("reset", &aeroforge::SimCore::reset)
        .def("step", &aeroforge::SimCore::step)
        .def("get_observation", [](aeroforge::SimCore& self) {
            constexpr int N = 13;  // [pos(3) + quat(4) + lin_vel(3) + ang_vel(3)]
            std::array<double, N> buffer{};
            self.getObservation(buffer.data(), N);
            // Return a NumPy array (copy)
            return py::array_t<double>(N, buffer.data());
        });
}
