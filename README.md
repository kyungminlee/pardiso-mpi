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

### Triplet-input example (SparseMatrixSolvePardiso)

```bash
mpirun -np 4 ./example_sparse_solve
```

All examples print the solution vector and the residual norm to verify correctness.

## Example Output

### `mpirun -np 4 ./example_direct`

```
Rank 0 solution:
  x[ 0] = 1.000000e+00  (exact = 1.000000e+00)
  x[ 1] = 1.000000e+00  (exact = 1.000000e+00)
  x[ 2] = 1.000000e+00  (exact = 1.000000e+00)
  x[ 3] = 1.000000e+00  (exact = 1.000000e+00)
Rank 1 solution:
  x[ 4] = 1.000000e+00  (exact = 1.000000e+00)
  x[ 5] = 1.000000e+00  (exact = 1.000000e+00)
  x[ 6] = 1.000000e+00  (exact = 1.000000e+00)
  x[ 7] = 1.000000e+00  (exact = 1.000000e+00)
Rank 2 solution:
  x[ 8] = 1.000000e+00  (exact = 1.000000e+00)
  x[ 9] = 1.000000e+00  (exact = 1.000000e+00)
  x[10] = 1.000000e+00  (exact = 1.000000e+00)
  x[11] = 1.000000e+00  (exact = 1.000000e+00)
Rank 3 solution:
  x[12] = 1.000000e+00  (exact = 1.000000e+00)
  x[13] = 1.000000e+00  (exact = 1.000000e+00)
  x[14] = 1.000000e+00  (exact = 1.000000e+00)
  x[15] = 1.000000e+00  (exact = 1.000000e+00)

Residual ||Ax - b||_2 = 1.332268e-15
Solution error ||x - x_exact||_2 = 4.577567e-16
```

### `mpirun -np 4 ./example_mmio examples/laplacian_16x16.mtx`

```
Reading matrix from examples/laplacian_16x16.mtx ...
Matrix: 16 x 16, nnz = 64
Rank 0 solution:
  x[ 0] = 8.333333e-01
  x[ 1] = 1.166667e+00
  x[ 2] = 1.166667e+00
  x[ 3] = 8.333333e-01
Rank 1 solution:
  x[ 4] = 1.166667e+00
  x[ 5] = 1.666667e+00
  x[ 6] = 1.666667e+00
  x[ 7] = 1.166667e+00
Rank 2 solution:
  x[ 8] = 1.166667e+00
  x[ 9] = 1.666667e+00
  x[10] = 1.666667e+00
  x[11] = 1.166667e+00
Rank 3 solution:
  x[12] = 8.333333e-01
  x[13] = 1.166667e+00
  x[14] = 1.166667e+00
  x[15] = 8.333333e-01

Residual ||Ax - b||_2 = 2.106500e-15
```

### `mpirun -np 4 ./example_sparse_solve`

```
Solution:
  x[ 0] = 1.000000e+00
  x[ 1] = 1.000000e+00
  x[ 2] = 1.000000e+00
  x[ 3] = 1.000000e+00
  x[ 4] = 1.000000e+00
  x[ 5] = 1.000000e+00
  x[ 6] = 1.000000e+00
  x[ 7] = 1.000000e+00
  x[ 8] = 1.000000e+00
  x[ 9] = 1.000000e+00
  x[10] = 1.000000e+00
  x[11] = 1.000000e+00
  x[12] = 1.000000e+00
  x[13] = 1.000000e+00
  x[14] = 1.000000e+00
  x[15] = 1.000000e+00

||x - x_exact||_2 = 4.577567e-16
```
