// example_sparse_solve.cpp
// Demonstrates SparseMatrixSolvePardiso with triplet input.
// Builds the same 16x16 2D Laplacian as example_direct but via triplets.
//
// Build and run:
//   mpirun -np 4 ./example_sparse_solve

#include "sparse_matrix_solve_pardiso.h"
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <vector>

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const int N  = 16;
    const int NX = 4;
    const int NY = 4;

    // Build triplets for 2D Laplacian (0-based indices).
    // Intentionally add the diagonal as two separate triplets to test
    // duplicate summation: 4.0 = 2.0 + 2.0.
    std::vector<Triplet> triplets;
    for (int i = 0; i < N; ++i) {
        int gx = i % NX;
        int gy = i / NX;

        // Diagonal — split into two entries to exercise duplicate handling.
        triplets.push_back({i, i, 2.0});
        triplets.push_back({i, i, 2.0});

        if (gx > 0)      triplets.push_back({i, i - 1,  -1.0});
        if (gx < NX - 1) triplets.push_back({i, i + 1,  -1.0});
        if (gy > 0)      triplets.push_back({i, i - NX, -1.0});
        if (gy < NY - 1) triplets.push_back({i, i + NX, -1.0});
    }

    // Build RHS: b = A * x_exact with x_exact = [1, ..., 1].
    std::vector<double> rhs(N, 0.0);
    for (auto const& t : triplets)
        rhs[t.row] += t.value;   // * 1.0

    // Solve. Scope the solver so it is destroyed before MPI_Finalize.
    std::vector<double> sol(N, 0.0);
    {
        SparseMatrixSolvePardiso solver(MPI_COMM_WORLD, N);
        solver.update(triplets);
        solver.solve(rhs.data(), sol.data());
    }

    // Print and verify on rank 0.
    if (rank == 0) {
        std::printf("Solution:\n");
        for (int i = 0; i < N; ++i)
            std::printf("  x[%2d] = %12.6e\n", i, sol[i]);

        double err = 0.0;
        for (int i = 0; i < N; ++i) {
            double d = sol[i] - 1.0;
            err += d * d;
        }
        std::printf("\n||x - x_exact||_2 = %e\n", std::sqrt(err));
    }

    MPI_Finalize();
    return 0;
}
