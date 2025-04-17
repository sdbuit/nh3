import numpy as np
from _nh3_cpp import grade_percent, vsp_kw_t

class OptimizedProcessor:
    def __init__(self, config):
        self.mass = getattr(config, 'VEHICLE_MASS', 1500.0)
        self.crr = getattr(config, 'ROLLING_RESISTANCE_COEF', 0.0135)
        self.cd = getattr(config, 'DRAG_COEF', 0.369)
        self.area = getattr(config, 'FRONTAL_AREA_M2', 2.2)
        self.eff = getattr(config, 'DRIVETRAIN_EFFICIENCY', 0.92)

    def calculate_grade(self, alt_np: np.ndarray, dist_np: np.ndarray) -> np.ndarray:
        return grade_percent(alt_np, dist_np)

    def calculate_vsp(self, speed_np, accel_np, grade_np):
        return vsp_kw_t(speed_np, accel_np, grade_np,
                        self.mass, self.crr, self.cd, self.area, self.eff)