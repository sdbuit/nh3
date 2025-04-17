import numpy as np
from . import _nh3_cpp

def fast_grade_calculation(latitudes, longitudes, altitudes,
                           distance_threshold=10.0, smoothing_window=5):
    """
    Fast implementation of grade calculation using the C++ extension.
    Calculates road grade (%) with smoothing applied.
    """
    lat_array = np.asarray(latitudes, dtype=np.float64)
    lon_array = np.asarray(longitudes, dtype=np.float64)
    alt_array = np.asarray(altitudes, dtype=np.float64)
    if lat_array.size == 0:
        return np.array([], dtype=np.float64)

    grades = _nh3_cpp.calculate_grade(
        lat_array, lon_array, alt_array,
        distance_threshold=distance_threshold,
        smoothing_window=smoothing_window
    )
    return grades

def calculate_vsp_from_data(speeds_kph, timestamps, grades,
                            vehicle_mass=1500.0,
                            rolling_resistance_coef=0.0135,
                            drag_coef=0.369,
                            frontal_area=2.27,
                            drivetrain_efficiency=0.92,
                            air_density=1.207):
    """
    Compute VSP (kW/tonne) using the C++ extension.
    Returns a tuple (vsp_values, accelerations).
    """
    speeds_array = np.asarray(speeds_kph, dtype=np.float64) / 3.6
    time_array = np.asarray(timestamps, dtype=np.float64)
    grade_array = np.asarray(grades, dtype=np.float64)
    vsp_vals, accels = _nh3_cpp.calculate_vsp(
        speeds_array, time_array, grade_array,
        vehicle_mass, rolling_resistance_coef, drag_coef, frontal_area,
        drivetrain_efficiency, air_density
    )
    return vsp_vals, accels

def classify_vsp_modes(vsp_values):
    """
    Classify VSP values into mode categories using the C++ extension.
    Returns an array of mode indices for each VSP value.
    """
    vsp_array = np.asarray(vsp_values, dtype=np.float64)
    modes = _nh3_cpp.classify_vsp_modes(vsp_array)
    return modes
