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

find_package(Eigen3 REQUIRED)

add_executable(fem_eigen
    fem_poisson_p1.cpp
)

target_link_libraries(fem_eigen
    Eigen3::Eigen
)

target_compile_options(fem_eigen PRIVATE
    -O3
    -march=native
)
