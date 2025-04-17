#define _USE_MATH_DEFINES
#include <cmath> // Now M_PI should be defined

#include "grade_calc.h"
#include <algorithm> // For std::max, std::min
#include <numeric>
#include <stdexcept> // For std::invalid_argument
#include <vector>    // Ensure vector is included

constexpr double EARTH_RADIUS = 6371000.0; // meters
constexpr double MIN_GRADE = -30.0;        // percent
constexpr double MAX_GRADE = 30.0;         // percent

namespace emissions_analysis
{

    inline double deg2rad(double deg)
    {
        return deg * M_PI / 180.0;
    }

    std::vector<double> calculate_distances(
        const std::vector<double> &latitudes,
        const std::vector<double> &longitudes)
    {
        const size_t n = latitudes.size();
        if (longitudes.size() != n)
        {
            throw std::invalid_argument("Latitude and longitude vectors must have the same size in calculate_distances");
        }
        if (n == 0)
        {
            return {}; // Handle empty input
        }

        std::vector<double> distances(n, 0.0);
        // distances[0] is already 0.0;

        // Calculate distances using Haversine formula
        for (size_t i = 1; i < n; ++i)
        {
            const double lat1 = deg2rad(latitudes[i - 1]);
            const double lon1 = deg2rad(longitudes[i - 1]);
            const double lat2 = deg2rad(latitudes[i]);
            const double lon2 = deg2rad(longitudes[i]);

            // Haversine formula
            const double dlat = lat2 - lat1;
            const double dlon = lon2 - lon1;
            const double a = std::sin(dlat / 2.0) * std::sin(dlat / 2.0) +
                             std::cos(lat1) * std::cos(lat2) *
                                 std::sin(dlon / 2.0) * std::sin(dlon / 2.0);
            // Add epsilon to prevent sqrt(0) issues in atan2 for identical points
            const double epsilon = 1e-12;
            const double c = 2.0 * std::atan2(std::sqrt(a), std::sqrt(1.0 - a + epsilon));
            distances[i] = EARTH_RADIUS * c;
        }
        return distances;
    }

    // Apply moving average smoothing.
    // Consider Savitzky-Golay or other methods for different smoothing characteristics.
    std::vector<double> smooth_values(
        const std::vector<double> &values, int window_size)
    {
        const size_t n = values.size();
        if (n == 0)
        {
            return {}; // Handle empty input
        }
        std::vector<double> smoothed(n, 0.0);

        // Ensure window size is odd and at least 3 for a symmetrical window centered on the point
        if (window_size <= 1)
        {
            return values; // No smoothing needed or invalid window
        }
        if (window_size % 2 == 0)
        {
            window_size += 1; // Force odd
        }
        window_size = std::max(3, window_size); // Ensure minimum size of 3
        const int half_window = window_size / 2;

        // Moving average (consider optimizing with a sliding window sum for large datasets)
        for (size_t i = 0; i < n; ++i)
        {
            double sum = 0.0;
            int count = 0;
            // Use signed int for index calculation, but check bounds against size_t n
            for (int j = -half_window; j <= half_window; ++j)
            {
                const int current_idx = static_cast<int>(i) + j; // Cast i to int for calculation
                // Check bounds carefully
                if (current_idx >= 0 && static_cast<size_t>(current_idx) < n)
                {
                    sum += values[static_cast<size_t>(current_idx)];
                    count++;
                }
            }
            smoothed[i] = (count > 0) ? sum / count : values[i]; // Avoid division by zero
        }
        return smoothed;
    }

    std::vector<double> calculate_grade(
        const std::vector<double> &latitudes,
        const std::vector<double> &longitudes,
        const std::vector<double> &altitudes,
        double distance_threshold,
        int smoothing_window)
    {
        const size_t n = latitudes.size();
        // Ensure all input vectors have the same size
        if (longitudes.size() != n || altitudes.size() != n)
        {
            throw std::invalid_argument("Input vectors must have the same size in calculate_grade");
        }
        if (n == 0)
        {
            return {}; // Handle empty input
        }

        std::vector<double> distances = calculate_distances(latitudes, longitudes);
        std::vector<double> grades(n, 0.0);
        // grades[0] is already 0.0;
        for (size_t i = 1; i < n; ++i)
        {
            const double distance = distances[i];
            // Only calculate grade if horizontal distance is meaningful
            if (distance > distance_threshold)
            {
                const double elevation_change = altitudes[i] - altitudes[i - 1];
                // Calculate raw grade, protect against division by zero/very small number
                grades[i] = (std::abs(distance) > 1e-6) ? (elevation_change / distance) * 100.0 : 0.0;
            }
            else
            {
                // If the horizontal distance is too small, assume grade is zero.
                // Assigning the previous grade (as done previously) could propagate errors.
                grades[i] = 0.0;
            }
            // Limit extreme grades (potentially due to GPS errors) using defined constants
            grades[i] = std::max(MIN_GRADE, std::min(MAX_GRADE, grades[i]));
            // Removed contradictory comment about not using GPS elevation
        }

        // Smooth the calculated grades
        std::vector<double> smoothed_grades = smooth_values(grades, smoothing_window);

        return smoothed_grades;
    }
}
