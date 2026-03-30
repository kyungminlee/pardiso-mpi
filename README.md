# pardiso-mpi

A C++ wrapper around Intel MKL PARDISO for distributed-memory (MPI) sparse direct solves. Each MPI rank owns a subset of rows; the wrapper coordinates the symbolic/numeric factorization and solve phases across ranks.

## Requirements

- CMake 3.15+
- C++17 compiler
- MPI implementation (OpenMPI, MPICH, Intel MPI, etc.)
- Intel MKL (oneAPI or standalone)

## Build

```bash
mkdir build && cd build
cmake ..
make -j
```

If CMake cannot find MKL automatically, set the `MKLROOT` environment variable:

```bash
export MKLROOT=/opt/intel/oneapi/mkl/latest
cmake ..
```

## Usage

### Direct example (hardcoded 16x16 Laplacian)

```bash
mpirun -np 4 ./example_direct
```

### Matrix Market example

```bash
mpirun -np 4 ./example_mmio ../examples/laplacian_16x16.mtx
```

Both examples print the solution vector and the residual norm to verify correctness.
