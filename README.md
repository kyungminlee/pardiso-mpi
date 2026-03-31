# pardiso-mpi

A C++ wrapper around Intel MKL **Cluster PARDISO** for MPI-parallel sparse
direct solves.  Cluster PARDISO distributes the factorization and solve
work across all MPI ranks internally — no manual data partitioning is
required.

The solver classes are templated on the scalar type — `double` or
`std::complex<double>` — so the same API works for both real and complex
systems.

## Requirements

- CMake 3.15+
- C++17 compiler
- MPI implementation (OpenMPI, MPICH, Intel MPI, etc.)
- Intel MKL (oneAPI or standalone) with Cluster PARDISO support

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

### Two API layers

| Class | Input format | Use when... |
|-------|-------------|-------------|
| `PardisoMPI<Scalar>` | 1-based CSR | You already have CSR arrays |
| `SparseMatrixSolvePardiso<Scalar>` | 0-based COO triplets | You have (row, col, value) entries |

`Scalar` is `double` or `std::complex<double>`.  The default PARDISO
matrix type is chosen automatically (`11` for real, `13` for complex).

Both are collective — every MPI rank must call the same methods in the
same order.  Every rank supplies the same (identical) matrix and RHS.
After a solve, every rank holds the full solution.

### Quick start (PardisoMPI)

```cpp
#include "pardiso_mpi.h"

// All ranks have the same CSR matrix (ia, ja, a) and RHS (b).
PardisoMPI<double> solver(MPI_COMM_WORLD);
solver.set_matrix(n, local_nrows, owned_rows, ia, ja, a);
solver.factorize();                // symbolic + numeric
solver.solve(b, x);               // x has the full solution on every rank
```

Symbolic and numeric factorization can be invoked separately:

```cpp
solver.set_matrix(n, local_nrows, owned_rows, ia, ja, a);
solver.symbolic_factorize();       // phase 11 only
solver.numeric_factorize();        // phase 22 only
solver.solve(b, x);
```

Transpose and adjoint solves:

```cpp
solver.solve(b, x, PardisoMPI<double>::SolveType::transpose);  // A^T x = b
solver.solve(b, x, PardisoMPI<double>::SolveType::adjoint);    // A^H x = b
```

### Quick start (SparseMatrixSolvePardiso)

```cpp
#include "sparse_matrix_solve_pardiso.h"

// All ranks have the same triplets and RHS.
SparseMatrixSolvePardiso<double> solver(MPI_COMM_WORLD, n);
solver.update(triplets);           // COO -> CSR, factorize
solver.solve(rhs, sol);            // solve, broadcast solution
```

### Complex example

```cpp
#include "sparse_matrix_solve_pardiso.h"
#include <complex>

using cx = std::complex<double>;

SparseMatrixSolvePardiso<cx> solver(MPI_COMM_WORLD, n);
solver.set_triplets(triplets);     // Triplet<cx> entries
solver.symbolic_factorize();
solver.numeric_factorize();
solver.solve(rhs, sol);
```

### Examples

```bash
mpirun -np 4 ./example_direct         # Hardcoded 16x16 Laplacian (CSR)
mpirun -np 4 ./example_sparse_solve   # Same matrix via triplets
mpirun -np 4 ./example_complex        # Complex shifted Laplacian
mpirun -np 4 ./example_mmio ../examples/laplacian_16x16.mtx  # Matrix Market file
```

All examples print the solution vector and the residual norm to verify
correctness.

## Example Output

### `mpirun -np 4 ./example_direct`

```
Solution:
  x[ 0] = 1.000000e+00  (exact = 1.000000e+00)
  x[ 1] = 1.000000e+00  (exact = 1.000000e+00)
  ...
  x[15] = 1.000000e+00  (exact = 1.000000e+00)

Residual ||Ax - b||_2 = 1.332268e-15
Solution error ||x - x_exact||_2 = 4.577567e-16
```

### `mpirun -np 4 ./example_sparse_solve`

```
Solution:
  x[ 0] = 1.000000e+00
  ...
  x[15] = 1.000000e+00

||x - x_exact||_2 = 4.577567e-16
```

### `mpirun -np 4 ./example_complex`

```
Solution (complex shifted Laplacian):
  x[ 0] = (1.000000e+00, -5.465713e-17)
  x[ 1] = (1.000000e+00, -1.445328e-16)
  x[ 2] = (1.000000e+00, -1.055744e-16)
  ...
  x[14] = (1.000000e+00, -1.443042e-16)
  x[15] = (1.000000e+00, -5.465713e-17)

||x - x_exact||_2 = 1.168640e-15
```
