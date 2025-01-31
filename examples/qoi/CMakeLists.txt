cmake_minimum_required(VERSION 3.9)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

project(mgard_standard)

find_package(MGARD REQUIRED)
add_definitions(-DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -Wfatal-errors -Wall -Wextra")

add_executable(average average.cpp)
add_executable(average3d average3d.cpp)
add_executable(average3d_block average3d_block.cpp)
#add_executable(average3d_block_omp average3d_block_omp.cpp)

target_link_libraries(average mgard::mgard)
target_link_libraries(average3d mgard::mgard)
target_link_libraries(average3d_block mgard::mgard)
#target_link_libraries(average3d_block_omp mgard::mgard)
#find_package(OpenMP)
#if (OpenMP_FOUND)
#   message("OpenMP Found.")
#  target_link_libraries(average3d_block_omp mgard::mgard OpenMP::OpenMP_CXX)
#else()
#    message("OpenMP not found.")
#endif ()