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

// Build a 16x16 2D Laplacian matrix (5-point stencil on a 4x4 grid).
//
// Grid node numbering (row-major):
//
//   0  1  2  3
//   4  5  6  7
//   8  9 10 11
//  12 13 14 15
//
// For each node i at grid position (gy, gx) with gx = i%4, gy = i/4:
//   A(i,i) = 4.0              (diagonal)
//   A(i,j) = -1.0  if j is a direct neighbor (left, right, up, down)
//
// CSR arrays use 1-based indexing as required by PARDISO.
static CSRMatrix build_laplacian_16() {
    const int N  = 16; // matrix dimension
    const int NX = 4;  // grid columns
    const int NY = 4;  // grid rows

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
    CSRMatrix mat;
    mat.n   = N;
    mat.m   = N;
    mat.nnz = nnz;
    mat.ia.resize(N + 1);
    mat.ja.resize(nnz);
    mat.a.resize(nnz);

    // Second pass: fill CSR arrays.  Column indices within each row are sorted
    // in ascending order, which PARDISO requires.
    mat.ia[0] = 1; // 1-based
    int pos = 0;
    for (int i = 0; i < N; ++i) {
        int gx = i % NX;
        int gy = i / NX;

        // Collect (column, value) pairs for this row, then sort by column.
        struct Entry { int col; double val; };
        std::vector<Entry> entries;

        // Up neighbor (gy - 1)
        if (gy > 0) entries.push_back({i - NX, -1.0});
        // Left neighbor (gx - 1)
        if (gx > 0) entries.push_back({i - 1, -1.0});
        // Diagonal
        entries.push_back({i, 4.0});
        // Right neighbor (gx + 1)
        if (gx < NX - 1) entries.push_back({i + 1, -1.0});
        // Down neighbor (gy + 1)
        if (gy < NY - 1) entries.push_back({i + NX, -1.0});

        // Entries are already sorted because of the order we added them
        // (column indices increase: i-NX < i-1 < i < i+1 < i+NX).
        for (auto& e : entries) {
            mat.ja[pos] = e.col + 1; // convert to 1-based
            mat.a[pos]  = e.val;
            ++pos;
        }
        mat.ia[i + 1] = mat.ia[i] + static_cast<int>(entries.size());
    }

    return mat;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ----------------------------------------------------------------
    // 1. Build the 16x16 Laplacian matrix (every rank has the full matrix).
    // ----------------------------------------------------------------
    CSRMatrix A = build_laplacian_16();
    const int N = A.n;

    // ----------------------------------------------------------------
    // 2. Create row-to-rank mapping: distribute rows as evenly as possible.
    //    With 4 ranks and 16 rows: rank 0 -> rows 0-3, rank 1 -> rows 4-7, etc.
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
    // 3. Set up the right-hand side: b = A * x_exact, where x_exact = 1.
    //    This way the exact solution is the all-ones vector, making
    //    verification easy.
    // ----------------------------------------------------------------
    std::vector<double> x_exact(N, 1.0);
    std::vector<double> b(N, 0.0);

    // Compute b = A * x_exact using the CSR structure.
    for (int i = 0; i < N; ++i) {
        double sum = 0.0;
        for (int k = A.ia[i] - 1; k < A.ia[i + 1] - 1; ++k) {
            int j = A.ja[k] - 1; // 0-based column
            sum += A.a[k] * x_exact[j];
        }
        b[i] = sum;
    }

    // ----------------------------------------------------------------
    // 4. Solve the system using the PardisoMPI wrapper.
    // ----------------------------------------------------------------
    std::vector<double> x(N, 0.0);

    PardisoMPI solver;
    solver.set_matrix_type(PardisoMPI::REAL_NONSYMMETRIC); // mtype = 11
    solver.set_message_level(0);                           // no PARDISO output

    int error = solver.solve(A, b.data(), x.data(), row_to_rank, MPI_COMM_WORLD);

    if (error != 0) {
        if (rank == 0)
            std::fprintf(stderr, "PardisoMPI solve failed with error %d\n", error);
        MPI_Finalize();
        return 1;
    }

    // ----------------------------------------------------------------
    // 5. Each rank prints its portion of the solution.
    // ----------------------------------------------------------------
    for (int r = 0; r < size; ++r) {
        if (rank == r) {
            std::printf("Rank %d solution:\n", rank);
            for (int i = 0; i < N; ++i) {
                if (row_to_rank[i] == rank) {
                    std::printf("  x[%2d] = %12.6e  (exact = %12.6e)\n",
                                i, x[i], x_exact[i]);
                }
            }
            std::fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // ----------------------------------------------------------------
    // 6. Verify: compute residual ||A*x - b||_2 on rank 0.
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

        double err = 0.0;
        for (int i = 0; i < N; ++i) {
            double d = x[i] - x_exact[i];
            err += d * d;
        }
        err = std::sqrt(err);
        std::printf("Solution error ||x - x_exact||_2 = %e\n", err);
    }

    MPI_Finalize();
    return 0;
}
