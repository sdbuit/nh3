#pragma once



#include "nanoflann.hpp"

#include <vector>

class ElevKDTree {
public:
    ElevKDTree(pybind11::array_t<double> lats,
               pybind11::array_t<double> lons,
               pybind11::array_t<double> elev);

    pybind11::array_t<double> query(pybind11::array_t<double> qlat,
                                    pybind11::array_t<double> qlon) const;
private:
    struct Point { double x, y; };
    std::vector<Point> pts_;
    const pybind11::buffer_info lat_, lon_, elev_;

    struct Adaptor;
    using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<double, Adaptor>, Adaptor, 2>;

    std::unique_ptr<Adaptor> adaptor_;
    std::unique_ptr<KDTree>  index_;
};