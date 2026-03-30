// example_mmio.cpp
// Reads a Matrix Market file, distributes across MPI ranks, and solves
// using the PardisoMPI wrapper.
//
// Build and run:
//   mpirun -np 4 ./example_mmio ../examples/laplacian_16x16.mtx

#include "pardiso_mpi.h"
#include "mmio.h"
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Broadcast a CSRMatrix from rank 0 to all other ranks.
static void bcast_csr(CSRMatrix& mat, int root, MPI_Comm comm) {
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

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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
    // 2. Rank 0 reads the Matrix Market file.
    // ----------------------------------------------------------------
    CSRMatrix A;
    if (rank == 0) {
        std::printf("Reading matrix from %s ...\n", mtx_file);
        A = read_matrix_market(mtx_file);
        std::printf("Matrix: %d x %d, nnz = %d\n", A.n, A.m, A.nnz);
    }

    // ----------------------------------------------------------------
    // 3. Broadcast the matrix to all ranks.
    // ----------------------------------------------------------------
    bcast_csr(A, 0, MPI_COMM_WORLD);
    const int N = A.n;

    // ----------------------------------------------------------------
    // 4. Create row-to-rank mapping (even distribution).
    // ----------------------------------------------------------------
    std::vector<int> row_to_rank(N);
    {
        int rows_per_rank = N / size;
        int remainder     = N % size;
        int offset = 0;
        for (int r = 0; r < size; ++r) {
            int count = rows_per_rank + (r < remainder ? 1 : 0);
            for (int j = 0; j < count; ++j) {
                row_to_rank[offset + j] = r;
            }
            offset += count;
        }
    }

    // ----------------------------------------------------------------
    // 5. Set up RHS: b = [1, 1, ..., 1].
    // ----------------------------------------------------------------
    std::vector<double> b(N, 1.0);
    std::vector<double> x(N, 0.0);

    // ----------------------------------------------------------------
    // 6. Solve with PardisoMPI.
    // ----------------------------------------------------------------
    PardisoMPI solver;
    solver.set_matrix_type(PardisoMPI::REAL_NONSYMMETRIC);
    solver.set_message_level(0);

    int error = solver.solve(A, b.data(), x.data(), row_to_rank, MPI_COMM_WORLD);

    if (error != 0) {
        if (rank == 0)
            std::fprintf(stderr, "PardisoMPI solve failed with error %d\n", error);
        MPI_Finalize();
        return 1;
    }

    // ----------------------------------------------------------------
    // 7. Print solution (each rank prints its rows).
    // ----------------------------------------------------------------
    for (int r = 0; r < size; ++r) {
        if (rank == r) {
            std::printf("Rank %d solution:\n", rank);
            for (int i = 0; i < N; ++i) {
                if (row_to_rank[i] == rank) {
                    std::printf("  x[%2d] = %12.6e\n", i, x[i]);
                }
            }
            std::fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // ----------------------------------------------------------------
    // 8. Compute and print residual ||Ax - b||_2 on rank 0.
    // ----------------------------------------------------------------
    if (rank == 0) {
        std::vector<double> r_vec(N, 0.0);
        for (int i = 0; i < N; ++i) {
            double sum = 0.0;
            for (int k = A.ia[i] - 1; k < A.ia[i + 1] - 1; ++k) {
                int j = A.ja[k] - 1;
                sum += A.a[k] * x[j];
            }
            r_vec[i] = sum - b[i];
        }

        double norm = 0.0;
        for (int i = 0; i < N; ++i) norm += r_vec[i] * r_vec[i];
        norm = std::sqrt(norm);
        std::printf("\nResidual ||Ax - b||_2 = %e\n", norm);
    }

    MPI_Finalize();
    return 0;
}
