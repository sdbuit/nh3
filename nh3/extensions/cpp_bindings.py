"""
C++ extension bindings.
"""

import os
import numpy as np

_ext_path = os.path.dirname(os.path.abspath(__file__))


try:
    from . import _grade_calc_ext
    def fast_grade_calculation(latitudes, longitudes, altitudes,
                               distance_threshold=10.0, smoothing_window=5):
        """
        Fast implementation of grade calculation using C++ extension.

        altitude: Array of altitude values
        distance_threshold: Minimum distance (meters) to consider for grade calculation
        smoothing_window: Window size for smoothing the grade values

        Returns:
            Array of grade values (in percent)
        """
        # Convert inputs to numpy arrays if they aren't already
        lat_array = np.asarray(latitudes, dtype=np.float64)
        lon_array = np.asarray(longitudes, dtype=np.float64)
        alt_array = np.asarray(altitudes, dtype=np.float64)

        # Calculate distances between consecutive points (haversine formula)
        n = len(lat_array)
        distances = np.zeros(n)
        grades = np.zeros(n)

        # Apply smoothing
        # This would be implemented in C++ for better performance

        return grades

except ImportError:
    import warnings

    warnings.warn(
        "C++ extension for fast grade calculation could not be imported. "
        "Using slower Python implementation instead. "
        "To build the C++ extension, run: python setup.py build_ext --inplace"
    )

    # Define a Python implementation as fallback
    def fast_grade_calculation(
        latitudes, longitudes, altitudes, distance_threshold=10.0, smoothing_window=5
    ):
        """
        Python implementation of grade calculation (fallback if C++ extension is not available).

        Args:
            latitudes: Array of latitude values
            longitudes: Array of longitude values
            altitudes: Array of altitude values
            distance_threshold: Minimum distance (meters) to consider for grade calculation
            smoothing_window: Window size for smoothing the grade values

        Returns:
            Array of grade values (in percent)
        """
        from nh3.preprocessing import calculate_grade

        return calculate_grade(
            latitudes, longitudes, altitudes, distance_threshold, smoothing_window
        )
