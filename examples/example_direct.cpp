// example_direct.cpp
// Direct example: solves a 16x16 2D Laplacian system (4x4 grid, 5-point stencil)
// using the PardisoMPI wrapper with a hardcoded sparse matrix in CSR format.
//
// Build and run:
//   mpirun -np 4 ./example_direct

#include "pardiso_mpi.h"
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <vector>

// ---------------------------------------------------------------------------
// Build a 16x16 2D Laplacian matrix (5-point stencil on a 4x4 grid).
//
// Grid node numbering (row-major, 0-based):
//
//    0  1  2  3
//    4  5  6  7
//    8  9 10 11
//   12 13 14 15
//
// For each node i at grid position (gy, gx) with gx = i%4, gy = i/4:
//   A(i,i) = 4.0              (diagonal)
//   A(i,j) = -1.0  if j is a direct neighbor (left, right, up, down)
//
// No wrap-around: boundary nodes have fewer neighbors.
//
// Non-zeros per node type:
//   Corner nodes (0,3,12,15):  diagonal + 2 neighbors = 3 entries
//   Edge nodes (1,2,4,7,8,11,13,14): diagonal + 3 neighbors = 4 entries
//   Interior nodes (5,6,9,10): diagonal + 4 neighbors = 5 entries
//   Total nnz = 4*3 + 8*4 + 4*5 = 64
//
// CSR arrays use 1-based indexing as required by PARDISO.
// ---------------------------------------------------------------------------
static void build_laplacian_16(int& n, std::vector<int>& ia,
                               std::vector<int>& ja, std::vector<double>& a)
{
    const int N  = 16; // matrix dimension
    const int NX = 4;  // grid columns
    const int NY = 4;  // grid rows

    n = N;

    // First pass: count non-zeros per row so we can size the arrays.
    std::vector<int> row_nnz(N, 0);
    for (int i = 0; i < N; ++i) {
        int gx = i % NX;
        int gy = i / NX;
        row_nnz[i] = 1; // diagonal
        if (gx > 0)      ++row_nnz[i]; // left neighbor
        if (gx < NX - 1) ++row_nnz[i]; // right neighbor
        if (gy > 0)      ++row_nnz[i]; // up neighbor
        if (gy < NY - 1) ++row_nnz[i]; // down neighbor
    }

    int nnz = 0;
    for (int i = 0; i < N; ++i) nnz += row_nnz[i];

    // Allocate CSR arrays (1-based indexing).
    ia.resize(N + 1);
    ja.resize(nnz);
    a.resize(nnz);

    // Second pass: fill CSR arrays.  Column indices within each row are
    // inserted in ascending order (up, left, diagonal, right, down),
    // which satisfies PARDISO's requirement for sorted column indices.
    ia[0] = 1; // 1-based row pointer start
    int pos = 0;
    for (int i = 0; i < N; ++i) {
        int gx = i % NX;
        int gy = i / NX;

        // Up neighbor (gy - 1): column = i - NX
        if (gy > 0) {
            ja[pos] = (i - NX) + 1; // 1-based
            a[pos]  = -1.0;
            ++pos;
        }
        // Left neighbor (gx - 1): column = i - 1
        if (gx > 0) {
            ja[pos] = (i - 1) + 1;
            a[pos]  = -1.0;
            ++pos;
        }
        // Diagonal: column = i
        ja[pos] = i + 1;
        a[pos]  = 4.0;
        ++pos;
        // Right neighbor (gx + 1): column = i + 1
        if (gx < NX - 1) {
            ja[pos] = (i + 1) + 1;
            a[pos]  = -1.0;
            ++pos;
        }
        // Down neighbor (gy + 1): column = i + NX
        if (gy < NY - 1) {
            ja[pos] = (i + NX) + 1;
            a[pos]  = -1.0;
            ++pos;
        }

        ia[i + 1] = ia[i] + row_nnz[i];
    }
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // ----------------------------------------------------------------
    // 1. Build the 16x16 Laplacian matrix (every rank has the full
    //    matrix so that set_matrix() can extract the locally-owned rows).
    // ----------------------------------------------------------------
    int N = 0;
    std::vector<int>    ia, ja;
    std::vector<double> a;
    build_laplacian_16(N, ia, ja, a);

    // ----------------------------------------------------------------
    // 2. Create row-to-rank mapping: distribute rows as evenly as
    //    possible.  With 4 ranks and 16 rows:
    //      rank 0 -> rows 0-3
    //      rank 1 -> rows 4-7
    //      rank 2 -> rows 8-11
    //      rank 3 -> rows 12-15
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
    // 3. Set up the right-hand side.
    //    We choose x_exact = [1, 1, ..., 1] and compute b = A * x_exact
    //    so that we can verify the solution against a known answer.
    // ----------------------------------------------------------------
    std::vector<double> x_exact(N, 1.0);
    std::vector<double> b_global(N, 0.0);

    // b = A * x_exact  (using the full CSR structure)
    for (int i = 0; i < N; ++i) {
        double sum = 0.0;
        for (int k = ia[i] - 1; k < ia[i + 1] - 1; ++k) {
            int col = ja[k] - 1; // convert to 0-based column
            sum += a[k] * x_exact[col];
        }
        b_global[i] = sum;
    }

    // Extract the local portion of b for this rank.
    std::vector<int> local_rows;
    for (int i = 0; i < N; ++i) {
        if (row_to_rank[i] == rank)
            local_rows.push_back(i);
    }
    const int local_n = static_cast<int>(local_rows.size());

    std::vector<double> b_local(local_n);
    for (int li = 0; li < local_n; ++li) {
        b_local[li] = b_global[local_rows[li]];
    }

    // ----------------------------------------------------------------
    // 4. Solve the system using the PardisoMPI wrapper.
    //    The solver must be destroyed before MPI_Finalize, so we scope it.
    // ----------------------------------------------------------------
    std::vector<double> x_local(local_n, 0.0);
    {
        PardisoMPI solver(MPI_COMM_WORLD);
        solver.set_matrix_type(11); // real unsymmetric

        // Provide the full CSR matrix and row ownership map; the wrapper
        // extracts only the locally-owned rows on each rank.
        solver.set_matrix(N, ia.data(), ja.data(), a.data(), row_to_rank.data());
        solver.factorize();
        solver.solve(b_local.data(), x_local.data());
    }

    // ----------------------------------------------------------------
    // 5. Each rank prints its portion of the solution.
    // ----------------------------------------------------------------
    for (int r = 0; r < nprocs; ++r) {
        if (rank == r) {
            std::printf("Rank %d solution:\n", rank);
            for (int li = 0; li < local_n; ++li) {
                int gi = local_rows[li];
                std::printf("  x[%2d] = %12.6e  (exact = %12.6e)\n",
                            gi, x_local[li], x_exact[gi]);
            }
            std::fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // ----------------------------------------------------------------
    // 6. Verify: gather full solution on rank 0 and compute the
    //    residual ||A*x - b||_2.
    // ----------------------------------------------------------------

    // Gather full solution onto rank 0 for residual computation.
    // Each rank sends its local_n values; rank 0 reassembles.
    std::vector<int> recv_counts(nprocs), recv_displs(nprocs);
    MPI_Gather(&local_n, 1, MPI_INT, recv_counts.data(), 1, MPI_INT,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        recv_displs[0] = 0;
        for (int r = 1; r < nprocs; ++r)
            recv_displs[r] = recv_displs[r - 1] + recv_counts[r - 1];
    }

    // Gather local row indices
    std::vector<int> all_rows(rank == 0 ? N : 0);
    MPI_Gatherv(local_rows.data(), local_n, MPI_INT,
                all_rows.data(), recv_counts.data(), recv_displs.data(),
                MPI_INT, 0, MPI_COMM_WORLD);

    // Gather local solution values
    std::vector<double> all_x_flat(rank == 0 ? N : 0);
    MPI_Gatherv(x_local.data(), local_n, MPI_DOUBLE,
                all_x_flat.data(), recv_counts.data(), recv_displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Reassemble global solution in order.
        std::vector<double> x_global(N, 0.0);
        for (int k = 0; k < N; ++k) {
            x_global[all_rows[k]] = all_x_flat[k];
        }

        // Compute residual r = A*x - b.
        std::vector<double> residual(N, 0.0);
        for (int i = 0; i < N; ++i) {
            double sum = 0.0;
            for (int k = ia[i] - 1; k < ia[i + 1] - 1; ++k) {
                int col = ja[k] - 1;
                sum += a[k] * x_global[col];
            }
            residual[i] = sum - b_global[i];
        }

        double res_norm = 0.0;
        for (int i = 0; i < N; ++i) res_norm += residual[i] * residual[i];
        res_norm = std::sqrt(res_norm);
        std::printf("\nResidual ||Ax - b||_2 = %e\n", res_norm);

        double err_norm = 0.0;
        for (int i = 0; i < N; ++i) {
            double d = x_global[i] - x_exact[i];
            err_norm += d * d;
        }
        err_norm = std::sqrt(err_norm);
        std::printf("Solution error ||x - x_exact||_2 = %e\n", err_norm);
    }

    MPI_Finalize();
    return 0;
}
