cmake_minimum_required(VERSION 3.16)
project(fem_poisson LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(cnl INTERFACE)
target_include_directories(cnl INTERFACE /opt/cnl/include)
add_library(cnl::cnl ALIAS cnl)

add_executable(fem_example csr_problem_reader.cpp fem_poisson_p1.cpp)

target_link_libraries(fem_example PRIVATE cnl::cnl)

target_compile_options(fem_example PRIVATE -O3)