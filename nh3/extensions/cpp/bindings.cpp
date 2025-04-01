#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "grade_calc.h"

namespace py = pybind11;

PYBIND11_MODULE(_grade_calc_ext, m) {
    m.doc() = "C++ extensions";

    m.def("calculate_grade",
          [](py::array_t<double> latitudes,
             py::array_t<double> longitudes,
             py::array_t<double> altitudes,
             double distance_threshold,
             int smoothing_window) {

              // Convert numpy arrays to std::vector
              auto lat_buf = latitudes.request();
              auto lon_buf = longitudes.request();
              auto alt_buf = altitudes.request();

              if (lat_buf.ndim != 1 || lon_buf.ndim != 1 || alt_buf.ndim != 1)
                  throw std::runtime_error("Input must be 1-dimensional arrays");

              if (lat_buf.size != lon_buf.size || lat_buf.size != alt_buf.size)
                  throw std::runtime_error("Input arrays must have the same size");

              const double* lat_ptr = static_cast<double*>(lat_buf.ptr);
              const double* lon_ptr = static_cast<double*>(lon_buf.ptr);
              const double* alt_ptr = static_cast<double*>(alt_buf.ptr);

              std::vector<double> lat_vec(lat_ptr, lat_ptr + lat_buf.size);
              std::vector<double> lon_vec(lon_ptr, lon_ptr + lon_buf.size);
              std::vector<double> alt_vec(alt_ptr, alt_ptr + alt_buf.size);

              // Call the C++ function
              std::vector<double> result = emissions_analysis::calculate_grade(
                  lat_vec, lon_vec, alt_vec, distance_threshold, smoothing_window);

              // Convert result back to numpy array
              auto result_array = py::array_t<double>(result.size());
              auto result_buf = result_array.request();
              double* result_ptr = static_cast<double*>(result_buf.ptr);

              for (size_t i = 0; i < result.size(); ++i) {
                  result_ptr[i] = result[i];
              }

              return result_array;
          },
          "Calculate grade (slope) values from GPS coordinates and altitude data",
          py::arg("latitudes"),
          py::arg("longitudes"),
          py::arg("altitudes"),
          py::arg("distance_threshold") = 10.0,
          py::arg("smoothing_window") = 5);

    m.def("calculate_distances",
          [](py::array_t<double> latitudes,
             py::array_t<double> longitudes) {

              // Convert numpy arrays to std::vector
              auto lat_buf = latitudes.request();
              auto lon_buf = longitudes.request();

              if (lat_buf.ndim != 1 || lon_buf.ndim != 1)
                  throw std::runtime_error("Input must be 1-dimensional arrays");

              if (lat_buf.size != lon_buf.size)
                  throw std::runtime_error("Input arrays must have the same size");

              const double* lat_ptr = static_cast<double*>(lat_buf.ptr);
              const double* lon_ptr = static_cast<double*>(lon_buf.ptr);

              std::vector<double> lat_vec(lat_ptr, lat_ptr + lat_buf.size);
              std::vector<double> lon_vec(lon_ptr, lon_ptr + lon_buf.size);

              std::vector<double> result = emissions_analysis::calculate_distances(lat_vec, lon_vec);

              // Convert result back to numpy array
              auto result_array = py::array_t<double>(result.size());
              auto result_buf = result_array.request();
              double* result_ptr = static_cast<double*>(result_buf.ptr);

              for (size_t i = 0; i < result.size(); ++i) {
                  result_ptr[i] = result[i];
              }

              return result_array;
          },
          "Calculate distances between consecutive GPS points using the Haversine formula",
          py::arg("latitudes"),
          py::arg("longitudes"));

    m.def("smooth_values",
          [](py::array_t<double> values,
             int window_size) {
              // Convert numpy array to std::vector
              auto val_buf = values.request();
              if (val_buf.ndim != 1)
                  throw std::runtime_error("Input must be a 1-dimensional array");

              const double* val_ptr = static_cast<double*>(val_buf.ptr);
              std::vector<double> val_vec(val_ptr, val_ptr + val_buf.size);
              std::vector<double> result = emissions_analysis::smooth_values(val_vec, window_size);

              // Convert result back to numpy array
              auto result_array = py::array_t<double>(result.size());
              auto result_buf = result_array.request();
              double* result_ptr = static_cast<double*>(result_buf.ptr);

              for (size_t i = 0; i < result.size(); ++i) {
                  result_ptr[i] = result[i];
              }

              return result_array;
          },
          "Apply smoothing to a vector of values",
          py::arg("values"),
          py::arg("window_size") = 5);
}
