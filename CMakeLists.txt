cmake_minimum_required(VERSION 3.15)
project(_grade_calc_ext LANGUAGES CXX)

# Add pybind11 from submodule
add_subdirectory(extern/pybind11)

# Define the extension module
pybind11_add_module(_grade_calc_ext
    nh3/extensions/cpp/bindings.cpp
    nh3/extensions/cpp/grade_calc.cpp
)

target_include_directories(_grade_calc_ext PRIVATE
    nh3/extensions/cpp
)


# Set .pyd output to nh3/extensions for ALL configurations (Debug/Release)
foreach(OUTPUTCONFIG IN LISTS CMAKE_CONFIGURATION_TYPES)
    string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG_UPPER)
    set_target_properties(_grade_calc_ext PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UPPER} ${CMAKE_CURRENT_SOURCE_DIR}/nh3/extensions
        LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UPPER} ${CMAKE_CURRENT_SOURCE_DIR}/nh3/extensions
        ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UPPER} ${CMAKE_CURRENT_SOURCE_DIR}/nh3/extensions
    )
endforeach()

# Ensure filename is clean (no lib prefix or extra suffix)
set_target_properties(_grade_calc_ext PROPERTIES
    PREFIX ""
    SUFFIX ".pyd"
)
