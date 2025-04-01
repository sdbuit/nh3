// C++ header file for fast grade calculation.
// Defines the interface for the C++ implementation of grade calculation.

#ifndef GRADE_CALC_H
#define GRADE_CALC_H

#include <vector>
#include <cmath>

namespace emissions_analysis {

/**
 * Calculate grade (slope) values from GPS coordinates and altitude data.
 *
 * @param latitudes Vector of latitude values in degrees
 * @param longitudes Vector of longitude values in degrees
 * @param altitudes Vector of altitude values in meters
 * @param distance_threshold Minimum distance (meters) to consider for grade calculation
 * @param smoothing_window Window size for smoothing the grade values
 * @return Vector of grade values (in percent)
 */
std::vector<double> calculate_grade(
    const std::vector<double>& latitudes,
    const std::vector<double>& longitudes,
    const std::vector<double>& altitudes,
    double distance_threshold = 10.0,
    int smoothing_window = 5
);

/**
 * Calculate distances between consecutive points using the Haversine formula.
 *
 * @param latitudes Vector of latitude values in degrees
 * @param longitudes Vector of longitude values in degrees
 * @return Vector of distances in meters
 */
std::vector<double> calculate_distances(
    const std::vector<double>& latitudes,
    const std::vector<double>& longitudes
);

/**
 * Apply Savitzky-Golay smoothing to a vector of values.
 *
 * @param values Vector of values to smooth
 * @param window_size Window size for smoothing (must be odd)
 * @return Vector of smoothed values
 */
std::vector<double> smooth_values(
    const std::vector<double>& values,
    int window_size
);

} // namespace emissions_analysis

#endif // GRADE_CALC_H
