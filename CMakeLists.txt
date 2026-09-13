cmake_minimum_required(VERSION 3.27)

project(ParallelTopologyOptimization)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
    ${CMAKE_BINARY_DIR}/bin
)

if(NOT CMAKE_BUILD_TYPE)
    set(
        CMAKE_BUILD_TYPE Release
        CACHE STRING "Build type" FORCE
    )
endif()
# ============================
# C++ standard
# ============================

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# ============================
# compile_commands.json
# ============================

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# ============================
# deal.II
# ============================
set(CMAKE_PREFIX_PATH "/home/xjtukkl/workspace/dealii/deal.II-v9.8.0/lib/cmake/deal.II")
find_package(deal.II 9.8 REQUIRED
    HINTS ${DEAL_II_DIR}
)

deal_ii_initialize_cached_variables()


# ============================
# Add sub-projects
# ============================


# Elasticity

add_subdirectory(
    "elasticity/Structural Optimization"
)

add_subdirectory(
    "elasticity/Structural Inverse Homogenization"
)


# Navier-Stokes

add_subdirectory(
    "navier-stokes/NS Flow Thermal Optimization"
)


# Thermal

add_subdirectory(
    "thermal/MatrixFree Optimization"
)

add_subdirectory(
    "thermal/MatrixFree Inverse Homogenization"
)

add_subdirectory(
    "thermal/Transient Phase Change Material"
)