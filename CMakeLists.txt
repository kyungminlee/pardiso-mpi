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

    # Cluster PARDISO requires layered linking — libmkl_rt does not
    # expose cluster_sparse_solver.
    find_library(MKL_INTEL_LP64 mkl_intel_lp64
        PATHS "${MKLROOT}/lib" "${MKLROOT}/lib/intel64"
              "${MKLROOT}/lib/x86_64-linux-gnu")
    find_library(MKL_SEQUENTIAL mkl_sequential
        PATHS "${MKLROOT}/lib" "${MKLROOT}/lib/intel64"
              "${MKLROOT}/lib/x86_64-linux-gnu")
    find_library(MKL_CORE mkl_core
        PATHS "${MKLROOT}/lib" "${MKLROOT}/lib/intel64"
              "${MKLROOT}/lib/x86_64-linux-gnu")

    # Pick the BLACS variant that matches the detected MPI implementation.
    # Default to Intel MPI / MPICH ABI; switch to OpenMPI when detected.
    set(_blacs_name "mkl_blacs_intelmpi_lp64")
    if(MPI_CXX_LIBRARIES MATCHES "openmpi" OR MPI_C_LIBRARIES MATCHES "openmpi"
       OR MPI_CXX_COMPILER MATCHES "openmpi" OR MPI_C_COMPILER MATCHES "openmpi"
       OR MPI_CXX_INCLUDE_DIRS MATCHES "openmpi" OR MPI_C_INCLUDE_DIRS MATCHES "openmpi")
        set(_blacs_name "mkl_blacs_openmpi_lp64")
    endif()
    message(STATUS "Using MKL BLACS variant: ${_blacs_name}")
    find_library(MKL_BLACS ${_blacs_name}
        PATHS "${MKLROOT}/lib" "${MKLROOT}/lib/intel64"
              "${MKLROOT}/lib/x86_64-linux-gnu")

    set(MKL_LIBRARIES ${MKL_INTEL_LP64} ${MKL_SEQUENTIAL} ${MKL_CORE})
    if(MKL_BLACS)
        list(APPEND MKL_LIBRARIES ${MKL_BLACS})
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
    # Use -Wl,--no-as-needed to ensure all MKL libraries (especially BLACS)
    # are linked even when the linker does not see direct references to them.
    # Cluster PARDISO loads BLACS symbols at runtime via libmkl_core.
    target_link_libraries(pardiso_mpi PUBLIC
        -Wl,--no-as-needed ${MKL_LIBRARIES} -Wl,--as-needed
        pthread m dl)
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

add_executable(example_complex examples/example_complex.cpp)
target_link_libraries(example_complex PRIVATE pardiso_mpi)
