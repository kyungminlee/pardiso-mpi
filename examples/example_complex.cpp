// example_complex.cpp
// Demonstrates SparseMatrixSolvePardiso<std::complex<double>> with triplet
// input.  Builds a complex 16x16 shifted Laplacian: A = L + 0.5i * I,
// where L is the 2D Laplacian (5-point stencil on a 4x4 grid) and I is
// the identity.  The imaginary shift makes the matrix complex unsymmetric.
//
// Build and run:
//   mpirun -np 4 ./example_complex

#include "sparse_matrix_solve_pardiso.h"
#include <mpi.h>
#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

using cx = std::complex<double>;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const int N  = 16;
    const int NX = 4;
    const int NY = 4;

    // Build triplets for complex shifted Laplacian (0-based indices).
    // Diagonal: 4.0 + 0.5i   Off-diagonal: -1.0 + 0.0i
    std::vector<Triplet<cx>> triplets;
    for (int i = 0; i < N; ++i) {
        int gx = i % NX;
        int gy = i / NX;

        // Diagonal with imaginary shift.
        triplets.push_back({i, i, cx(4.0, 0.5)});

        if (gx > 0)      triplets.push_back({i, i - 1,  cx(-1.0, 0.0)});
        if (gx < NX - 1) triplets.push_back({i, i + 1,  cx(-1.0, 0.0)});
        if (gy > 0)      triplets.push_back({i, i - NX, cx(-1.0, 0.0)});
        if (gy < NY - 1) triplets.push_back({i, i + NX, cx(-1.0, 0.0)});
    }

    // Build RHS: b = A * x_exact with x_exact = [1+0i, 1+0i, ..., 1+0i].
    const cx one(1.0, 0.0);
    std::vector<cx> rhs(N, cx(0.0, 0.0));
    for (auto const& t : triplets)
        rhs[t.row] += t.value * one;

    // Solve. Scope the solver so it is destroyed before MPI_Finalize.
    std::vector<cx> sol(N, cx(0.0, 0.0));
    {
        SparseMatrixSolvePardiso<cx> solver(MPI_COMM_WORLD, N);

        // Use separate factorization phases to demonstrate the new API.
        solver.set_triplets(triplets);
        solver.symbolic_factorize();
        solver.numeric_factorize();
        solver.solve(rhs.data(), sol.data());
    }

    // Print and verify on rank 0.
    if (rank == 0) {
        std::printf("Solution (complex shifted Laplacian):\n");
        for (int i = 0; i < N; ++i)
            std::printf("  x[%2d] = (%12.6e, %12.6e)\n",
                        i, sol[i].real(), sol[i].imag());

        double err = 0.0;
        for (int i = 0; i < N; ++i) {
            cx d = sol[i] - one;
            err += std::norm(d);   // |d|^2
        }
        std::printf("\n||x - x_exact||_2 = %e\n", std::sqrt(err));
    }

    MPI_Finalize();
    return 0;
}
