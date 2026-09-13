
# Set the name of the project and target:
set(TARGET "HeatFlowTopOpt")

file(GLOB_RECURSE TARGET_SRC  "src/*.cc")
file(GLOB_RECURSE TARGET_INC  "include/*.h")
set(TARGET_SRC ${TARGET_SRC}  ${TARGET_INC})


# Usually, you will not need to modify anything beyond this point...

cmake_minimum_required(VERSION 3.13.4)
set(CMAKE_PREFIX_PATH "/home/xjtukkl/workspace/dealii/deal.II-v9.8.0/lib/cmake/deal.II")
find_package(deal.II 9.8.0 REQUIRED HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR})

# Are all dependencies fulfilled?

if(NOT DEAL_II_WITH_PETSC) # keep in one line
  message(FATAL_ERROR "
Error! This tutorial requires a deal.II library that was configured with the following options:
    DEAL_II_WITH_PETSC = ON
However, the deal.II library found at ${DEAL_II_PATH} was configured with these options:
    DEAL_II_WITH_PETSC = ${DEAL_II_WITH_PETSC}
This conflicts with the requirements."
    )
endif()


# For MPI Include package
find_package(MPI REQUIRED)
include_directories (${MPI_CXX_INCLUDE_DIRS})

# Assignment CMAKE_CXX_COMPILER
set(CMAKE_CXX_COMPILER, "/usr/bin/mpicxx")

# Debug or Release
set(CMAKE_BUILD_TYPE "Release")

deal_ii_initialize_cached_variables()
project(${TARGET})
deal_ii_invoke_autopilot()
