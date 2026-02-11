cmake_minimum_required(VERSION 3.16)

project(fem_poisson
    VERSION 0.1
    LANGUAGES CXX
)

# C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(fem_example
    fem_poisson_p1.cpp
)

target_compile_options(fem_example PRIVATE
    -O3
    -march=native
)
