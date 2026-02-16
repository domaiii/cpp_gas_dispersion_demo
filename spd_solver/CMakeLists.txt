cmake_minimum_required(VERSION 3.16)
project(spd_solver)

set(CMAKE_CXX_STANDARD 20)

add_library(spd_solver
    src/csr.cpp
)

target_include_directories(spd_solver PUBLIC src)

add_subdirectory(tests)
add_subdirectory(solver)