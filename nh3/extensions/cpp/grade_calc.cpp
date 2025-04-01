// C++ implementation of fast grade calculation.
#include "grade_calc.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace emissions_analysis {

const double EARTH_RADIUS = 6371000.0;

inline double deg2rad(double deg) {
    return deg * M_PI / 180.0;
}

std::vector<double> calculate_distances(const std::vector<double>& latitudes,
    const std::vector<double>& longitudes) {
        const size_t n = latitudes.size();
        std::vector<double> distances(n, 0.0);
    distances[0] = 0.0;
    // Calculate distances using Haversine formula
    for (size_t i = 1; i < n; ++i) {
        const double lat1 = deg2rad(latitudes[i-1]);
        const double lon1 = deg2rad(longitudes[i-1]);
        const double lat2 = deg2rad(latitudes[i]);
        const double lon2 = deg2rad(longitudes[i]);

        // Haversine formula
        const double dlat = lat2 - lat1;
        const double dlon = lon2 - lon1;
        const double a = std::sin(dlat/2) * std::sin(dlat/2) +
                         std::cos(lat1) * std::cos(lat2) *
                         std::sin(dlon/2) * std::sin(dlon/2);
        const double c = 2 * std::atan2(std::sqrt(a), std::sqrt(1-a));
        distances[i] = EARTH_RADIUS * c;
    }
    return distances;
}

std::vector<double> smooth_values(
    const std::vector<double>& values, int window_size) {
        const size_t n = values.size();
        std::vector<double> smoothed(n, 0.0);

    if (window_size % 2 == 0) {
        window_size += 1;}
    window_size = std::max(3, window_size);
    const int half_window = window_size / 2;

    // Moving average for now (could be replaced with Savitzky-Golay)
    for (size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        int count = 0;
        for (int j = -half_window; j <= half_window; ++j) {
            const int idx = i + j;
            if (idx >= 0 && idx < static_cast<int>(n)) {
                sum += values[idx];
                count++;
            }
        }
        smoothed[i] = (count > 0) ? sum / count : values[i];
    }
    return smoothed;
}

std::vector<double> calculate_grade(
    const std::vector<double>& latitudes,
    const std::vector<double>& longitudes,
    const std::vector<double>& altitudes,
    double distance_threshold,
    int smoothing_window) {
        const size_t n = latitudes.size();
    // Ensure all input vectors have the same size
    if (longitudes.size() != n || altitudes.size() != n) {
        throw std::invalid_argument("Input vectors must have the same size");}
    std::vector<double> distances = calculate_distances(latitudes, longitudes);
    std::vector<double> grades(n, 0.0);
    grades[0] = 0.0;
    // Calculate grades as rise/run * 100 (percentage)
    for (size_t i = 1; i < n; ++i) {
        const double distance = distances[i];
        // Only calculate grade if distance is above threshold
        if (distance > distance_threshold) {
            const double elevation_change = altitudes[i] - altitudes[i-1];
            grades[i] = (elevation_change / distance) * 100.0;
        } else {
            // For small distances, use the previous grade
            // NOTE: avoid division by small numbers
            grades[i] = (i > 1) ? grades[i-1] : 0.0;
        }
        // Limit extreme grades (likely due to GPS errors)
        // Also, GPS elevation will not be used for grade calutaltions.
        grades[i] = std::max(-30.0, std::min(30.0, grades[i]));
    }
    std::vector<double> smoothed_grades = smooth_values(grades,
        smoothing_window);

    return smoothed_grades;
}
}
