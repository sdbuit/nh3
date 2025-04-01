try:
    from extensions.cpp_bindings import fast_grade_calculation

    has_cpp_extensions = True
except ImportError:
    from nh3.preprocessing import calculate_grade as fast_grade_calculation

    has_cpp_extensions = False

__all__ = ["fast_grade_calculation", "has_cpp_extensions"]
