cmake_minimum_required(VERSION 3.27)

project(NSHeatFlowTopOpt)


# ============================
# Source files
# ============================

file(GLOB_RECURSE TARGET_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

file(GLOB_RECURSE TARGET_INC
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
)

# ============================
# Executable
# ============================

add_executable(
    ${PROJECT_NAME}
    ${TARGET_SRC}
    ${TARGET_INC}
)


# ============================
# Include
# ============================

target_include_directories(
    ${PROJECT_NAME}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)


# ============================
# deal.II setup
# ============================

deal_ii_setup_target(
    ${PROJECT_NAME}
)

# ============================
# Copy input files
# ============================

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/input)

    add_custom_command(
        TARGET ${PROJECT_NAME}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/input
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/input
    )

endif()