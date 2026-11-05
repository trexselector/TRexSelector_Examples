# =============================================================================
# TRexSelector_Examples — cpp/CMakeLists.txt
# =============================================================================
# Standalone CMake project for C++ demo and simulation executables.
# Consumes the installed TRexSelector library via find_package().
#
# Prerequisites:
#   cmake --install <TRexSelector-build-dir>
#   cmake -S cpp -B cpp/build/debug -DCMAKE_PREFIX_PATH=<prefix>
#
# The CMakePresets.json in this directory sets CMAKE_PREFIX_PATH to
# ../../TRexSelector-beyond/cpp/install by default.
# =============================================================================
cmake_minimum_required(VERSION 3.23...4.3)


# ============================================================================
# Compiler Selection (must match TRexSelector — Homebrew LLVM on macOS)
# ============================================================================
if(APPLE AND NOT DEFINED CMAKE_CXX_COMPILER)
    set(HOMEBREW_LLVM_PREFIX "/opt/homebrew/opt/llvm")
    if(EXISTS "${HOMEBREW_LLVM_PREFIX}/bin/clang++")
        set(CMAKE_C_COMPILER   "${HOMEBREW_LLVM_PREFIX}/bin/clang"   CACHE FILEPATH "C compiler")
        set(CMAKE_CXX_COMPILER "${HOMEBREW_LLVM_PREFIX}/bin/clang++" CACHE FILEPATH "C++ compiler")
        message(STATUS "*** Using Homebrew LLVM: ${CMAKE_CXX_COMPILER}")
    else()
        message(WARNING "*** Homebrew LLVM not found — falling back to system compiler")
    endif()
endif()


# ============================================================================
# Output Directories
# ============================================================================
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)


# ============================================================================
# Project Definition
# ============================================================================
project(TRexSelector_Examples
    VERSION 1.0.0
    DESCRIPTION "Demo and example executables for the TRexSelector library"
    LANGUAGES CXX
)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# ============================================================================
# Find TRexSelector (installed library)
# ============================================================================
find_package(TRexSelector REQUIRED)


# ============================================================================
# Find remaining dependencies
# ============================================================================

# OpenMP
if(APPLE)
    if(NOT DEFINED HOMEBREW_LLVM_PREFIX)
        set(HOMEBREW_LLVM_PREFIX "/opt/homebrew/opt/llvm")
    endif()
    set(OpenMP_CXX_FLAGS       "-fopenmp=libomp -I${HOMEBREW_LLVM_PREFIX}/include")
    set(OpenMP_CXX_LIB_NAMES   "libomp")
    set(OpenMP_libomp_LIBRARY  "${HOMEBREW_LLVM_PREFIX}/lib/libomp.dylib"
        CACHE FILEPATH "libomp path" FORCE)
endif()
find_package(OpenMP REQUIRED)

# Eigen3
find_package(Eigen3 REQUIRED NO_MODULE)

# Boost
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()
find_package(Boost 1.80 REQUIRED COMPONENTS headers)

# BLAS/LAPACK
if(APPLE)
    set(BLA_VENDOR Apple)
    find_package(BLAS)
    find_package(LAPACK)
    if(NOT BLAS_FOUND)
        set(BLA_VENDOR OpenBLAS)
        find_package(BLAS REQUIRED)
        find_package(LAPACK REQUIRED)
    endif()
else()
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
endif()

# Matplot++ (only needed here, no longer in the core library)
find_package(Matplot++ REQUIRED)


# ============================================================================
# Configuration Summary
# ============================================================================
message(STATUS "==============================================")
message(STATUS "TRexSelector_Examples Build Configuration")
message(STATUS "==============================================")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler:   ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "")
message(STATUS "Libraries:")
if(TRexSelector_FOUND)
    message(STATUS "  - TRexSelector: ${TRexSelector_VERSION} (${TRexSelector_DIR})")
else()
    message(STATUS "  - TRexSelector: NOT FOUND")
endif()
if(Matplot++_FOUND)
    message(STATUS "  - Matplot++: ${Matplot++_VERSION}")
else()
    message(STATUS "  - Matplot++: NOT FOUND")
endif()
if(Eigen3_FOUND)
    message(STATUS "  - Eigen3: ${Eigen3_VERSION}")
endif()
if(BLAS_FOUND)
    message(STATUS "  - BLAS Vendor: ${BLA_VENDOR}")
endif()
if(OpenMP_CXX_FOUND)
    message(STATUS "  - OpenMP: ${OpenMP_CXX_VERSION} (flags: ${OpenMP_CXX_FLAGS})")
else()
    message(STATUS "  - OpenMP: NOT FOUND")
endif()
message(STATUS "==============================================")


# ===================================================================================
# Library Sets
# ===================================================================================
# Base libraries for all demo executables.
set(BASE_LIBS
    Boost::headers
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

# Libraries for Data Preprocessing demos
set(DATA_PREPROCESSING_LIBS
    ${BASE_LIBS}
    TRexSelector::trex_ml_methods
)

# Libraries for T-Algorithm based demos
set(T_ALGORITHM_LIBS
    ${BASE_LIBS}
    TRexSelector::trex_ml_methods
    TRexSelector::trex_tsolvers
)

# Libraries for TRex Selector based demos
# (trex_selector_methods transitively pulls in tsolvers, ml_methods, utils)
set(T_REX_SELECTOR_LIBS
    ${BASE_LIBS}
    TRexSelector::trex_selector_methods
)


# ==================================================================================
# Macro: add_demo_target
# ==================================================================================
# Creates a demo executable with proper include paths, linking, and compile flags.
#
# Usage:
#   add_demo_target(<target_name> <demo_file> <library_set> [additional_libraries...])
#
# Arguments:
#   target_name          - Name of the demo executable (e.g., demo_ts_01_tlars)
#   demo_file            - Source file path relative to current CMakeLists.txt
#   library_set          - Library variable (DATA_PREPROCESSING_LIBS, T_ALGORITHM_LIBS,
#                          T_REX_SELECTOR_LIBS)
#   additional_libraries - (Optional) Additional libraries to link against
#
# ==================================================================================
macro(add_demo_target target_name demo_file library_set)

    add_executable(${target_name} ${demo_file})

    # Include directories: installed headers are carried via INTERFACE_INCLUDE_DIRECTORIES
    # on the TRexSelector:: targets. Additional paths can be injected via ADDITIONAL_INCLUDES.
    target_include_directories(${target_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${ADDITIONAL_INCLUDES}
    )

    target_link_libraries(${target_name} PRIVATE
        Eigen3::Eigen
        OpenMP::OpenMP_CXX
        ${${library_set}}
        ${ARGN}
    )

    # Route each demo's output into a simulation_results/data/ folder next to
    # its own source file (generated figures live in the sibling
    # simulation_results/plots/, written by the plotting scripts). Demos that
    # live directly in the group directory (no subfolder) keep the original
    # <group>/simulation_results/data/ location.
    get_filename_component(_demo_subdir "${demo_file}" DIRECTORY)
    if(_demo_subdir)
        set(_demo_output_dir "${CMAKE_CURRENT_SOURCE_DIR}/${_demo_subdir}/simulation_results/data/")
    else()
        set(_demo_output_dir "${CMAKE_CURRENT_SOURCE_DIR}/simulation_results/data/")
    endif()

    target_compile_definitions(${target_name} PRIVATE
        EIGEN_USE_BLAS
        DEMO_OUTPUT_DIR="${_demo_output_dir}"
    )

    # Mirror the source-tree layout under bin/ instead of dumping every demo
    # executable into one flat directory, e.g.:
    #   trex_selector_methods/ad/trex_core/demo_trex_01_single_run/demo_trex_01_single_run.cpp
    #     -> <build>/bin/trex_selector_methods/ad/trex_core/demo_trex_01_single_run/
    #   tsolvers/ad/demo_ts_01_tlars/demo_ts_01_tlars.cpp
    #     -> <build>/bin/tsolvers/ad/demo_ts_01_tlars/
    file(RELATIVE_PATH _group_rel_dir "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
    if(_demo_subdir)
        set(_demo_bin_dir "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${_group_rel_dir}/${_demo_subdir}")
    else()
        set(_demo_bin_dir "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${_group_rel_dir}")
    endif()
    set_target_properties(${target_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${_demo_bin_dir}"
    )

    # IEEE 754 conformant, as the library: no -ffast-math. The demos
    # instantiate header-inline library code (data generation, scalers,
    # templates), so their numbers stay bit-comparable with the library's own
    # builds and tests; -ffast-math also implies -ffinite-math-only, which
    # breaks std::isnan / std::isfinite.
    target_compile_options(${target_name} PRIVATE
        -Wno-deprecated-declarations
        $<$<CONFIG:Debug>:-g -O0 -Wall -Wextra>
        $<$<CONFIG:Release>:-O3 -DNDEBUG -march=native -funroll-loops>
    )

    set(ADDITIONAL_INCLUDES "")
endmacro()
# ==================================================================================


# ===================================================================================
# Demo/Simulation Targets
# ===================================================================================
add_subdirectory(memory_mapping)
add_subdirectory(ml_methods)
add_subdirectory(tsolvers)
add_subdirectory(trex_selector_methods)
