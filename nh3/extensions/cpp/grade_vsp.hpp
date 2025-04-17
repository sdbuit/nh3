#pragma once

#include "bindings.cpp"
#include "pybind11/numpy.h"
#include <cmath>
#include <omp.h>

constexpr double GRAVITY = 9.80665;
constexpr double AIR_RHO = 1.225;

inline pybind11::array_t<double> grade_percent(
    pybind11::array_t<double> alt,
    pybind11::array_t<double> dist)
{
    auto a = alt.unchecked<1>();
    auto d = dist.unchecked<1>();
    auto out = pybind11::array_t<double>(alt.size());
    auto o = out.mutable_unchecked<1>();

    o(0) = NAN;
    #pragma omp parallel for
    for (ssize_t i = 1; i < alt.size(); ++i) {
        double delta = d(i);
        o(i) = (delta > 0.01) ? (a(i) - a(i - 1)) / delta * 100.0 : NAN;
    }
    return out;
}

inline pybind11::array_t<double> vsp_kw_t(
    pybind11::array_t<double> speed,
    pybind11::array_t<double> accel,
    pybind11::array_t<double> grade_pct,
    double mass, double c_rr, double c_d, double frontal, double eta)
{
    auto v = speed.unchecked<1>();
    auto a = accel.unchecked<1>();
    auto g = grade_pct.unchecked<1>();
    auto out = pybind11::array_t<double>(speed.size());
    auto o = out.mutable_unchecked<1>();

    double roll = c_rr * GRAVITY;
    double drag = 0.5 * AIR_RHO * c_d * frontal / mass;

    #pragma omp parallel for
    for (ssize_t i = 0; i < speed.size(); ++i) {
        if (std::isnan(v(i)) || std::isnan(a(i)) || std::isnan(g(i))) {
            o(i) = NAN;
        } else {
            o(i) = (roll + drag * v(i)*v(i) + GRAVITY * (g(i)/100.0) + a(i)) * v(i) / eta;
        }
    }
    return out;
}