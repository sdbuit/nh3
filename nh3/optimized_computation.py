import numpy as np
from numba import jit, prange
from scipy import signal
import math


EARTH_RADIUS = 6371000.0
GRAVITY = 9.81


@jit(nopython=True)
def deg2rad(deg):
    return deg * math.pi / 180.0

@jit(nopython=True)
def haversine_distance_optimized(lat1, lon1, lat2, lon2):
    lat1_rad = deg2rad(lat1)
    lon1_rad = deg2rad(lon1)
    lat2_rad = deg2rad(lat2)
    lon2_rad = deg2rad(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS * c

@jit(nopython=True, parallel=True)
def calculate_distances_optimized(latitudes, longitudes):
    n = len(latitudes)
    distances = np.zeros(n)
    distances[0] = 0.0
    for i in prange(1, n):
        distances[i] = haversine_distance_optimized(
            latitudes[i - 1], longitudes[i - 1], latitudes[i], longitudes[i]
        )
    return distances

@jit(nopython=True)
def calculate_grade_optimized(
    latitudes, longitudes, altitudes, distance_threshold=10.0
):
    n = len(latitudes)
    distances = calculate_distances_optimized(latitudes, longitudes)
    grades = np.zeros(n)
    grades[0] = 0.0
    for i in range(1, n):
        distance = distances[i]
        if distance > distance_threshold:
            elevation_change = altitudes[i] - altitudes[i - 1]
            grades[i] = (elevation_change / distance) * 100.0
        else:
            grades[i] = grades[i - 1] if i > 1 else 0.0
        if grades[i] > 30.0:
            grades[i] = 30.0
        elif grades[i] < -30.0:
            grades[i] = -30.0
    return grades

def smooth_savgol(values, window_size=11, poly_order=2):
    if window_size % 2 == 0:
        window_size += 1
    window_size = max(3, window_size)
    poly_order = min(poly_order, window_size - 1)
    try:
        return signal.savgol_filter(values, window_size, poly_order)
    except Exception:
        return smooth_moving_average(values, window_size)

@jit(nopython=True)
def smooth_moving_average(values, window_size):
    n = len(values)
    smoothed = np.zeros(n)
    if window_size % 2 == 0:
        window_size += 1
    window_size = max(3, window_size)
    half_window = window_size // 2
    for i in range(n):
        window_start = max(0, i - half_window)
        window_end = min(n, i + half_window + 1)
        window_sum = 0.0
        count = 0
        for j in range(window_start, window_end):
            window_sum += values[j]
            count += 1
        smoothed[i] = window_sum / count if count > 0 else values[i]
    return smoothed

def calculate_grade_with_smoothing(
    latitudes, longitudes, altitudes, distance_threshold=10.0, smoothing_window=11
):
    raw_grades = calculate_grade_optimized(
        latitudes, longitudes, altitudes, distance_threshold
    )
    return smooth_savgol(raw_grades, smoothing_window)

@jit(nopython=True)
def calculate_accelerations_optimized(speeds, time_deltas):
    n = len(speeds)
    accelerations = np.zeros(n)
    if n > 1:
        accelerations[0] = (speeds[1] - speeds[0]) / max(time_deltas[0], 0.1)
    for i in range(1, n - 1):
        dt_prev = max(time_deltas[i - 1], 0.1)
        dt_next = max(time_deltas[i], 0.1)
        dt_total = dt_prev + dt_next
        accelerations[i] = (
            (speeds[i + 1] - speeds[i]) / dt_next * dt_prev
            + (speeds[i] - speeds[i - 1]) / dt_prev * dt_next
        ) / dt_total
    if n > 1:
        accelerations[n - 1] = (speeds[n - 1] - speeds[n - 2]) / max(
            time_deltas[n - 2], 0.1
        )
    for i in range(n):
        if accelerations[i] > 5.0:
            accelerations[i] = 5.0
        elif accelerations[i] < -5.0:
            accelerations[i] = -5.0
    return accelerations

@jit(nopython=True)
def calculate_vsp_optimized(
    speeds, accelerations, grades, rolling_term, drag_term, drivetrain_efficiency
):
    n = len(speeds)
    vsp = np.zeros(n)
    slopes = grades / 100.0
    for i in range(n):
        speed = speeds[i]
        acceleration = accelerations[i]
        slope = slopes[i]
        rolling_resistance = rolling_term * speed
        aerodynamic_drag = drag_term * speed**3
        grade_resistance = GRAVITY * slope * speed
        inertial_power = speed * acceleration
        vsp[i] = (
            rolling_resistance + aerodynamic_drag + grade_resistance + inertial_power
        ) / drivetrain_efficiency
    return vsp

@jit(nopython=True)
def calculate_time_deltas_optimized(timestamps):
    n = len(timestamps)
    time_deltas = np.zeros(n)
    if n > 1:
        time_deltas[0] = timestamps[1] - timestamps[0]
        for i in range(1, n - 1):
            time_deltas[i] = timestamps[i + 1] - timestamps[i]
        time_deltas[n - 1] = time_deltas[n - 2]
    for i in range(n):
        if time_deltas[i] <= 0:
            time_deltas[i] = 0.1
    return time_deltas

def calculate_vsp_from_data_optimized(
    speeds_kph,
    timestamps,
    grades,
    vehicle_mass=1500.0,
    rolling_resistance_coef=0.0135,
    drag_coef=0.369,
    frontal_area=2.27,
    drivetrain_efficiency=0.92,
    air_density=1.207,
):
    speeds_ms = speeds_kph / 3.6
    time_deltas = calculate_time_deltas_optimized(timestamps)
    accelerations = calculate_accelerations_optimized(speeds_ms, time_deltas)
    rolling_term = rolling_resistance_coef * GRAVITY
    drag_term = 0.5 * air_density * drag_coef * frontal_area / vehicle_mass
    vsp_values = calculate_vsp_optimized(
        speeds_ms, accelerations, grades, rolling_term, drag_term, drivetrain_efficiency
    )
    return vsp_values, accelerations

@jit(nopython=True)
def classify_vsp_modes_optimized(vsp_values):
    n = len(vsp_values)
    modes = np.zeros(n, dtype=np.int32)
    thresholds = np.array(
        [-2.0, 0.0, 1.0, 4.0, 7.0, 10.0, 13.0, 16.0, 19.0, 23.0, 28.0, 33.0, 39.0]
    )
    for i in range(n):
        vsp = vsp_values[i]
        mode = 0
        for j in range(len(thresholds)):
            if vsp >= thresholds[j]:
                mode += 1
            else:
                break
        modes[i] = mode
    return modes
