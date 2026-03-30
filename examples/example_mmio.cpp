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
        int rows_per_rank = N / nprocs;
        int remainder     = N % nprocs;
        int offset = 0;
        for (int r = 0; r < nprocs; ++r) {
            int count = rows_per_rank + (r < remainder ? 1 : 0);
            for (int j = 0; j < count; ++j) {
                row_to_rank[offset + j] = r;
            }
            offset += count;
        }
    }

    // ----------------------------------------------------------------
    // 5. Set up RHS: b = [1, 1, ..., 1].
    //    Extract the local portion for this rank.
    // ----------------------------------------------------------------
    std::vector<int> local_rows;
    for (int i = 0; i < N; ++i) {
        if (row_to_rank[i] == rank)
            local_rows.push_back(i);
    }
    const int local_n = static_cast<int>(local_rows.size());

    std::vector<double> b_global(N, 1.0);
    std::vector<double> b_local(local_n);
    for (int li = 0; li < local_n; ++li) {
        b_local[li] = b_global[local_rows[li]];
    }

    // ----------------------------------------------------------------
    // 6. Solve with PardisoMPI.
    // ----------------------------------------------------------------
    PardisoMPI solver(MPI_COMM_WORLD);
    solver.set_matrix_type(11); // real unsymmetric

    solver.set_matrix(N, A.ia.data(), A.ja.data(), A.a.data(),
                      row_to_rank.data());
    solver.factorize();

    std::vector<double> x_local(local_n, 0.0);
    solver.solve(b_local.data(), x_local.data());

    // ----------------------------------------------------------------
    // 7. Print solution (each rank prints its rows in order).
    // ----------------------------------------------------------------
    for (int r = 0; r < nprocs; ++r) {
        if (rank == r) {
            std::printf("Rank %d solution:\n", rank);
            for (int li = 0; li < local_n; ++li) {
                std::printf("  x[%2d] = %12.6e\n", local_rows[li], x_local[li]);
            }
            std::fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // ----------------------------------------------------------------
    // 8. Gather full solution on rank 0 and compute residual.
    // ----------------------------------------------------------------
    std::vector<int> recv_counts(nprocs), recv_displs(nprocs);
    MPI_Gather(&local_n, 1, MPI_INT, recv_counts.data(), 1, MPI_INT,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        recv_displs[0] = 0;
        for (int r = 1; r < nprocs; ++r)
            recv_displs[r] = recv_displs[r - 1] + recv_counts[r - 1];
    }

    std::vector<int> all_rows(rank == 0 ? N : 0);
    MPI_Gatherv(local_rows.data(), local_n, MPI_INT,
                all_rows.data(), recv_counts.data(), recv_displs.data(),
                MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<double> all_x_flat(rank == 0 ? N : 0);
    MPI_Gatherv(x_local.data(), local_n, MPI_DOUBLE,
                all_x_flat.data(), recv_counts.data(), recv_displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Reassemble global solution.
        std::vector<double> x_global(N, 0.0);
        for (int k = 0; k < N; ++k) {
            x_global[all_rows[k]] = all_x_flat[k];
        }

        // Compute residual r = A*x - b.
        std::vector<double> residual(N, 0.0);
        for (int i = 0; i < N; ++i) {
            double sum = 0.0;
            for (int k = A.ia[i] - 1; k < A.ia[i + 1] - 1; ++k) {
                int col = A.ja[k] - 1;
                sum += A.a[k] * x_global[col];
            }
            residual[i] = sum - b_global[i];
        }

        double res_norm = 0.0;
        for (int i = 0; i < N; ++i) res_norm += residual[i] * residual[i];
        res_norm = std::sqrt(res_norm);
        std::printf("\nResidual ||Ax - b||_2 = %e\n", res_norm);
    }

    MPI_Finalize();
    return 0;
}
