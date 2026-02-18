add_executable(main_solver main.cpp)
# target_compile_definitions(main_solver PRIVATE DEBUG_CG)
target_link_libraries(main_solver spd_solver)
