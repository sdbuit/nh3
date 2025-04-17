import logging
from typing import Optional

import polars as pl
import numpy as np


GRAVITY = 9.80665
MPH_TO_MPS = 0.44704
KMH_TO_MPS = 1000 / 3600

logger = logging.getLogger(__name__)


#pragma once
#include <nanoflann.hpp>
#include <pybind11/numpy.h>
#include <vector>

class ElevKDTree {
  public:
      pybind11::array_t<double> lons, pybind11::array_t<double> elev);

    try:
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
    std::unique_ptr<KDTree>  index_;};
        def __init__(self,
                     vehicle_mass_kg: float,
                     rolling_resistance_coeff: float, # Unitless
                     drag_coeff: float,               # Unitlessfrontal_area_m2: float,          # m^2
        drivetrain_efficiency: float = 0.92, # Unitless (typically 0.9-0.95)
        air_density_kg_m3: float = 1.225,  # kg/m^3 (at sea level, 15°C)
    ):
        if not (0 < drivetrain_efficiency <= 1):
             logger.warning(f'Drivetrain efficiency ({drivetrain_efficiency}) outside typical range (0, 1]. Using 0.92.')
             drivetrain_efficiency = 0.92
        self.vehicle_mass_kg = vehicle_mass_kg
        self.drivetrain_efficiency = drivetrain_efficiency
        self.air_density_kg_m3 = air_density_kg_m3
        self.rolling_term = rolling_resistance_coeff * GRAVITY
        self.drag_term = (0.5 * air_density_kg_m3 * drag_coeff * frontal_area_m2) / vehicle_mass_kg
        logger.info(f'VSPCalculator initialized: mass={vehicle_mass_kg:.1f} kg, Crr={rolling_resistance_coeff:.4f}, '
                    f'Cd={drag_coeff:.3f}, A={frontal_area_m2:.2f} m^2, eta={drivetrain_efficiency:.2f}, rho={air_density_kg_m3:.3f} kg/m^3')
        logger.info(f'Calculated rolling_term: {self.rolling_term:.4f} m/s^2')
        logger.info(f'Calculated drag_term: {self.drag_term:.6f} 1/m')

    def _calculate_acceleration(self, time_s_np: np.ndarray,
                                speed_mps_np: np.ndarray,
                                smooth_window: Optional[int] = 10) -> np.ndarray:
        """Computes acceleration w/ optional smoothing."""
        file_basename = 'Accel Calc'
        if len(time_s_np) != len(speed_mps_np):
            raise ValueError('Time/speed array length mismatch.')
        if len(time_s_np) < 2:
            return np.zeros_like(time_s_np)
        nan_time = np.isnan(time_s_np).sum()
        nan_speed = np.isnan(speed_mps_np).sum()
        if nan_time > 0:
            logger.warning(f'[{file_basename}] Input time_s_np contains {nan_time} NaNs.')
        if nan_speed > 0:
            logger.warning(f'[{file_basename}] Input speed_mps_np contains {nan_speed} NaNs.')
        if smooth_window is not None and smooth_window > 1:
            logger.debug(f'[{file_basename}] Smoothing speed window={smooth_window}')
            window = np.ones(smooth_window) / smooth_window
            smoothed_speed_mps = np.convolve(speed_mps_np, window, mode='same')
            nan_smooth = np.isnan(smoothed_speed_mps).sum()
            if nan_smooth > nan_speed:
                logger.warning(f'[{file_basename}] Smoothing increased NaN count in speed to {nan_smooth}.')
        else:
            smoothed_speed_mps = speed_mps_np
        dt = np.diff(time_s_np)
        dv = np.diff(smoothed_speed_mps)
        if len(time_s_np) > 1:
            dt_0 = time_s_np[1] - time_s_np[0]
            dv_0 = smoothed_speed_mps[1] - smoothed_speed_mps[0]
            dt = np.concatenate(([dt_0], dt))
            dv = np.concatenate(([dv_0], dv))
        else:
            dt = np.array([1.0])
            dv = np.array([0.0])
        epsilon = 1e-6
        acceleration_mps2 = np.full_like(time_s_np, np.nan, dtype=float)
        valid_dt_mask = (np.abs(dt) > epsilon) & ~np.isnan(dt) & ~np.isnan(dv)
        acceleration_mps2[valid_dt_mask] = dv[valid_dt_mask] / dt[valid_dt_mask]
        nan_accel = np.isnan(acceleration_mps2).sum()
        if nan_accel > 0:
            logger.warning(f'[{file_basename}] Acceleration calculation resulted in {nan_accel} NaNs.')
        logger.debug('Finished acceleration calculation.')
        return acceleration_mps2

    def calculate_vsp(self, df: pl.DataFrame,
                      smooth_speed_window: Optional[int] = 10) -> np.ndarray:
        """Calculates VSP (kW/tonne)"""
        file_basename = 'VSP Calc'
        logger.info(f'[{file_basename}] Starting VSP calculation...')
        # Define Column Names
        time_col_name = 'time_s'
        grade_col_name = 'grade_percent'
        speed_col_options = {'speed_mph', 'speed_kmh'}
        # Input Validation
        required_cols = {time_col_name, grade_col_name}
        present_speed_cols = speed_col_options.intersection(df.columns)
        if not present_speed_cols or not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            if not present_speed_cols:
                missing.add('speed_mph or speed_kmh')
            logger.error(f'[{file_basename}] Missing required columns: {missing}. Cannot calculate VSP.')
            return np.full(len(df), np.nan)
        speed_col_in_name = 'speed_mph' if 'speed_mph' in present_speed_cols else 'speed_kmh'
        logger.info(f'[{file_basename}] Using input speed column: {speed_col_in_name}')
        try:
            logger.debug(f'[{file_basename}] Extracting data to np.arrays...')
            time_s_np = df[time_col_name].cast(pl.Float64, strict=False).to_numpy()
            speed_raw_np = df[speed_col_in_name].cast(pl.Float64, strict=False).to_numpy()
            grade_percent_np = df[grade_col_name].cast(pl.Float64, strict=False).to_numpy()
            logger.debug(f'[{file_basename}] Extracted array shapes: time={time_s_np.shape}, speed={speed_raw_np.shape}, grade={grade_percent_np.shape}')
        except Exception as e:
             logger.error(f'[{file_basename}] Error extracting data: {e}', exc_info=True)
             return np.full(len(df), np.nan)

        # Unit Conversions
        logger.debug(f'[{file_basename}] Performing unit conversions...')
        speed_mps_np = speed_raw_np * (MPH_TO_MPS if speed_col_in_name == 'speed_mph' else KMH_TO_MPS)
        grade_decimal_np = grade_percent_np / 100.0

        # Calculate Acceleration
        acceleration_mps2_np = self._calculate_acceleration(time_s_np, speed_mps_np)

        # Check for NaNs in inputs to VSP formula
        nan_speed_final = np.isnan(speed_mps_np).sum()
        nan_grade_final = np.isnan(grade_decimal_np).sum()
        nan_accel_final = np.isnan(acceleration_mps2_np).sum()
        if nan_speed_final > 0:
            logger.warning(f'[{file_basename}] Input speed_mps_np to VSP formula has {nan_speed_final} NaNs.')
        if nan_grade_final > 0:
            logger.warning(f'[{file_basename}] Input grade_decimal_np to VSP formula has {nan_grade_final} NaNs.')
        if nan_accel_final > 0:
            logger.warning(f'[{file_basename}] Input acceleration_mps2_np to VSP formula has {nan_accel_final} NaNs.')

        # Apply VSP Formula
        logger.debug(f'[{file_basename}] Applying VSP formula...')
        vsp_w_kg = np.full_like(time_s_np, np.nan, dtype=float) # Initialize with NaNs
        valid_mask = ( np.isfinite(speed_mps_np) & np.isfinite(grade_decimal_np) & np.isfinite(acceleration_mps2_np) )

        # Apply formula only for valid entries
        vsp_w_kg[valid_mask] = (
            (
                self.rolling_term + self.drag_term * np.square(speed_mps_np[valid_mask])
                + GRAVITY * grade_decimal_np[valid_mask]
                + acceleration_mps2_np[valid_mask]
            ) * speed_mps_np[valid_mask] / self.drivetrain_efficiency
        )
        vsp_kw_tonne = vsp_w_kg # W/kg is numerically equivalent to kW/tonne
        final_nan_count = np.isnan(vsp_kw_tonne).sum()
        logger.info(f'[{file_basename}] VSP calculation complete. Resulting NaNs: {final_nan_count} out of {len(vsp_kw_tonne)}.')

        return vsp_kw_tonne
    