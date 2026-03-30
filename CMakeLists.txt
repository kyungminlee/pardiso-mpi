cmake_minimum_required(VERSION 3.15)
project(pardiso-mpi LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --------------------------------------------------------------------------
# MPI
# --------------------------------------------------------------------------
find_package(MPI REQUIRED)

# --------------------------------------------------------------------------
# Intel MKL (provides PARDISO)
#
# Try MKLConfig.cmake first (ships with oneAPI / standalone MKL).
# Fall back to MKLROOT environment variable if the config is not found.
# --------------------------------------------------------------------------
find_package(MKL CONFIG QUIET)

if(MKL_FOUND)
    message(STATUS "Found MKL via MKLConfig.cmake")
    set(MKL_LIBRARIES MKL::MKL)
else()
    message(STATUS "MKLConfig.cmake not found -- falling back to MKLROOT")
    if(DEFINED ENV{MKLROOT})
        set(MKLROOT "$ENV{MKLROOT}")
    elseif(NOT DEFINED MKLROOT)
        message(FATAL_ERROR
            "MKL not found. Set MKLROOT environment variable or install oneAPI MKL.")
    endif()

    # Look for mkl.h in MKLROOT/include and MKLROOT/include/mkl
    set(MKL_INCLUDE_DIR "${MKLROOT}/include")
    if(NOT EXISTS "${MKL_INCLUDE_DIR}/mkl.h" AND EXISTS "${MKL_INCLUDE_DIR}/mkl/mkl.h")
        set(MKL_INCLUDE_DIR "${MKLROOT}/include/mkl")
    endif()

    # Prefer the single-dynamic-library link model (libmkl_rt).
    find_library(MKL_RT_LIB mkl_rt
        PATHS "${MKLROOT}/lib" "${MKLROOT}/lib/intel64"
              "${MKLROOT}/lib/x86_64-linux-gnu")

    if(MKL_RT_LIB)
        set(MKL_LIBRARIES ${MKL_RT_LIB})
    else()
        # Static / layered linking as fallback.
        # Cluster PARDISO requires BLACS in addition to the standard MKL libs.
        find_library(MKL_INTEL_LP64 mkl_intel_lp64
            PATHS "${MKLROOT}/lib" "${MKLROOT}/lib/intel64"
                  "${MKLROOT}/lib/x86_64-linux-gnu")
        find_library(MKL_SEQUENTIAL mkl_sequential
            PATHS "${MKLROOT}/lib" "${MKLROOT}/lib/intel64"
                  "${MKLROOT}/lib/x86_64-linux-gnu")
        find_library(MKL_CORE mkl_core
            PATHS "${MKLROOT}/lib" "${MKLROOT}/lib/intel64"
                  "${MKLROOT}/lib/x86_64-linux-gnu")

        # Try Intel MPI BLACS first, fall back to OpenMPI BLACS.
        find_library(MKL_BLACS mkl_blacs_intelmpi_lp64
            PATHS "${MKLROOT}/lib" "${MKLROOT}/lib/intel64"
                  "${MKLROOT}/lib/x86_64-linux-gnu")
        if(NOT MKL_BLACS)
            find_library(MKL_BLACS mkl_blacs_openmpi_lp64
                PATHS "${MKLROOT}/lib" "${MKLROOT}/lib/intel64"
                      "${MKLROOT}/lib/x86_64-linux-gnu")
        endif()

        set(MKL_LIBRARIES ${MKL_INTEL_LP64} ${MKL_SEQUENTIAL} ${MKL_CORE})
        if(MKL_BLACS)
            list(APPEND MKL_LIBRARIES ${MKL_BLACS})
        endif()
    endif()

    if(NOT EXISTS "${MKL_INCLUDE_DIR}/mkl.h")
        message(FATAL_ERROR "Cannot find mkl.h in ${MKL_INCLUDE_DIR}")
    endif()

    message(STATUS "Using MKL from MKLROOT=${MKLROOT}")
endif()

# --------------------------------------------------------------------------
# Library: pardiso_mpi
# --------------------------------------------------------------------------
add_library(pardiso_mpi
    src/pardiso_mpi.cpp
    src/mmio.cpp
    src/sparse_matrix_solve_pardiso.cpp
)

target_include_directories(pardiso_mpi
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# MPI
target_link_libraries(pardiso_mpi PUBLIC MPI::MPI_CXX)

# MKL
if(MKL_FOUND)
    target_link_libraries(pardiso_mpi PUBLIC ${MKL_LIBRARIES})
else()
    target_include_directories(pardiso_mpi PUBLIC ${MKL_INCLUDE_DIR})
    target_link_libraries(pardiso_mpi PUBLIC ${MKL_LIBRARIES} pthread m dl)
endif()

# --------------------------------------------------------------------------
# Examples
# --------------------------------------------------------------------------
add_executable(example_direct examples/example_direct.cpp)
target_link_libraries(example_direct PRIVATE pardiso_mpi)

add_executable(example_mmio examples/example_mmio.cpp)
target_link_libraries(example_mmio PRIVATE pardiso_mpi)

add_executable(example_sparse_solve examples/example_sparse_solve.cpp)
target_link_libraries(example_sparse_solve PRIVATE pardiso_mpi)
