cmake_minimum_required(VERSION 3.27)
find_package(MPI REQUIRED)
include_directories (${MPI_CXX_INCLUDE_DIRS})

find_package(deal.II 9.8.0 REQUIRED HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR})
set(CMAKE_CXX_COMPILER, "/usr/bin/mpicxx")

set(TARGET "MatrixFreeThermal")
deal_ii_initialize_cached_variables()

file(GLOB_RECURSE TARGET_SRC  "src/*.cpp")
file(GLOB_RECURSE TARGET_INC  "include/*.h")

set(TARGET_SRC ${TARGET_SRC}  ${TARGET_INC})
project(${TARGET})
deal_ii_invoke_autopilot()