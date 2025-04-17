try:
    from .cpp_bindings import fast_grade_calculation, calculate_vsp_from_data, classify_vsp_modes
except ImportError as e:
    raise ImportError('Required cpp extension not available.  Build and install the nh3 C++ extension.') from e

__all__ = ["fast_grade_calculation", "calculate_vsp_from_data", "classify_vsp_modes"]
