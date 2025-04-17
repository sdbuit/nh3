cmake_minimum_required(VERSION 3.27)
project(nh3 LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_nh3_cpp
    nh3/extensions/cpp/bindings.cpp
    nh3/extensions/cpp/improved_grade_calc.cpp
    nh3/extensions/cpp/improved_vsp_calc.cpp
)

target_include_directories(_nh3_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/nh3/extensions/cpp
)

set_target_properties(_nh3_cpp PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
)

target_compile_options(_nh3_cpp PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /openmp>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O3 -march=native -ffast-math -fopenmp>
)
