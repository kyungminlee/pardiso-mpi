// example_mmio.cpp
// Reads a Matrix Market file and solves using the PardisoMPI wrapper
// (Cluster PARDISO).
//
// Build and run:
//   mpirun -np 4 ./example_mmio ../examples/laplacian_16x16.mtx

#include "pardiso_mpi.h"
#include "mmio.h"
#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Broadcast a CSRMatrix from rank 0 to all other ranks.
static void bcast_csr(CSRMatrix& mat, int root, MPI_Comm comm)
{
    MPI_Bcast(&mat.n,   1, MPI_INT, root, comm);
    MPI_Bcast(&mat.m,   1, MPI_INT, root, comm);
    MPI_Bcast(&mat.nnz, 1, MPI_INT, root, comm);

    // Resize on non-root ranks.
    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank != root) {
        mat.ia.resize(mat.n + 1);
        mat.ja.resize(mat.nnz);
        mat.a.resize(mat.nnz);
    }

    MPI_Bcast(mat.ia.data(), mat.n + 1, MPI_INT,    root, comm);
    MPI_Bcast(mat.ja.data(), mat.nnz,   MPI_INT,    root, comm);
    MPI_Bcast(mat.a.data(),  mat.nnz,   MPI_DOUBLE, root, comm);
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // ----------------------------------------------------------------
    // 1. Parse command-line arguments.
    // ----------------------------------------------------------------
    if (argc < 2) {
        if (rank == 0)
            std::fprintf(stderr, "Usage: %s <matrix_market_file>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    const char* mtx_file = argv[1];

    // ----------------------------------------------------------------
    // 2. Rank 0 reads the Matrix Market file; broadcast to all ranks.
    // ----------------------------------------------------------------
    CSRMatrix A;
    if (rank == 0) {
        std::printf("Reading matrix from %s ...\n", mtx_file);
        A = read_matrix_market(mtx_file);
        std::printf("Matrix: %d x %d, nnz = %d\n", A.n, A.m, A.nnz);
    }
    bcast_csr(A, 0, MPI_COMM_WORLD);
    const int N = A.n;

    // ----------------------------------------------------------------
    // 3. Partition rows across MPI ranks and extract local CSR.
    // ----------------------------------------------------------------
    int rows_per_rank = N / nprocs;
    int remainder     = N % nprocs;
    int first_row_0   = rank * rows_per_rank + std::min(rank, remainder);
    int local_nrows   = rows_per_rank + (rank < remainder ? 1 : 0);

    std::vector<int> owned_rows(local_nrows);
    for (int i = 0; i < local_nrows; ++i)
        owned_rows[i] = first_row_0 + i + 1;   // 1-based

    int nnz_offset = A.ia[first_row_0] - 1;
    std::vector<int> local_ia(local_nrows + 1);
    for (int i = 0; i <= local_nrows; ++i)
        local_ia[i] = A.ia[first_row_0 + i] - A.ia[first_row_0] + 1;

    // ----------------------------------------------------------------
    // 4. Set up RHS: b = [1, 1, ..., 1].
    // ----------------------------------------------------------------
    std::vector<double> b(N, 1.0);

    // ----------------------------------------------------------------
    // 5. Solve with Cluster PARDISO.
    //    Scope the solver so it is destroyed before MPI_Finalize.
    // ----------------------------------------------------------------
    std::vector<double> x(N, 0.0);
    {
        PardisoMPI<double> solver(MPI_COMM_WORLD);
        solver.set_matrix_type(11); // real unsymmetric
        solver.set_matrix(N, local_nrows, owned_rows.data(),
                          local_ia.data(),
                          A.ja.data() + nnz_offset,
                          A.a.data() + nnz_offset);
        solver.factorize();
        solver.solve(b.data(), x.data());
    }

    // ----------------------------------------------------------------
    // 6. Print solution and verify on rank 0.
    // ----------------------------------------------------------------
    if (rank == 0) {
        std::printf("Solution:\n");
        for (int i = 0; i < N; ++i)
            std::printf("  x[%2d] = %12.6e\n", i, x[i]);

        // Compute residual r = A*x - b.
        std::vector<double> residual(N, 0.0);
        for (int i = 0; i < N; ++i) {
            double sum = 0.0;
            for (int k = A.ia[i] - 1; k < A.ia[i + 1] - 1; ++k) {
                int col = A.ja[k] - 1;
                sum += A.a[k] * x[col];
            }
            residual[i] = sum - b[i];
        }

        double res_norm = 0.0;
        for (int i = 0; i < N; ++i) res_norm += residual[i] * residual[i];
        res_norm = std::sqrt(res_norm);
        std::printf("\nResidual ||Ax - b||_2 = %e\n", res_norm);
    }

    MPI_Finalize();
    return 0;
}
