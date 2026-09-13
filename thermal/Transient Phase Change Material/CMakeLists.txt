cmake_minimum_required(VERSION 3.27)
find_package(MPI REQUIRED)
include_directories (${MPI_CXX_INCLUDE_DIRS})
set(CMAKE_PREFIX_PATH "/home/xjtukkl/workspace/dealii/deal.II-v9.8.0/lib/cmake/deal.II")
find_package(deal.II 9.8.0 REQUIRED HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR})
set(CMAKE_CXX_COMPILER, "/usr/bin/mpicxx")

set(TARGET "NSTopOpt")
deal_ii_initialize_cached_variables()

file(GLOB_RECURSE TARGET_SRC  "src/*.cpp")
file(GLOB_RECURSE TARGET_INC  "include/*.hpp")

set(TARGET_SRC ${TARGET_SRC}  ${TARGET_INC})
project(${TARGET})
deal_ii_invoke_autopilot()