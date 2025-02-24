#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "kd_tree.h"

namespace py = pybind11;

PYBIND11_MODULE(kdtree_module, m) {
    m.doc() = "Custom KD-tree implementation";
    // Expose a function from kd_tree
    m.def("build_tree", [](std::vector<Point> points) {
        return build_kdtree(points.begin(), points.end());
    });
    
    // More bindings can be added here (future reference).
}
