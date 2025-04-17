#include <pybind11/pybind11.h>

#include "C:/Users/sam/AppData/Local/Programs/Python/Python312/Lib/site-packages/pybind11/include"

#include "grade_vsp.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_nh3_cpp, m) {
    m.doc() = "High-performance grade and VSP calculator (C++)";

    m.def("grade_percent", &grade_percent, "Vectorized grade calculation");
    m.def("vsp_kw_t", &vsp_kw_t, "Vectorized VSP calculation",
        py::arg("speed"), py::arg("accel"), py::arg("grade_pct"),
        py::arg("mass"), py::arg("c_rr"), py::arg("c_d"), py::arg("frontal"),
        py::arg("eta") = 0.92);
}
