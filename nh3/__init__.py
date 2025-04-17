import importlib

try:
    importlib.import_module('nh3.extensions._nh3_cpp')
except ImportError as e:
    raise ImportError('nh3 requires the compiled C++ extension (_nh3_cpp) to be installed.') from e
