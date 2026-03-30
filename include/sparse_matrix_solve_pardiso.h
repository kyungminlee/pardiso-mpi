#pragma once

#include "pardiso_mpi.h"
#include <mpi.h>
#include <vector>

/// A sparse-matrix triplet (row, col, value) using 0-based indices.
struct Triplet {
    int row;
    int col;
    double value;
};

/// High-level MPI-parallel sparse solver wrapper around PardisoMPI.
///
/// Every rank must supply the **same** complete matrix (via triplets) and
/// the **same** complete RHS vector.  Cluster PARDISO distributes the
/// factorization and solve work across all MPI ranks internally.
/// The solution is broadcast so every rank holds the full result.
///
/// Usage:
///   SparseMatrixSolvePardiso solver(MPI_COMM_WORLD, n);
///   solver.update(triplets);              // provide matrix, factorize
///   solver.solve(rhs, sol);               // solve A*sol = rhs
class SparseMatrixSolvePardiso {
public:
    /// @param comm   MPI communicator.
    /// @param n      Global matrix dimension (rows = cols).
    /// @param mtype  PARDISO matrix type (default 11 = real unsymmetric).
    SparseMatrixSolvePardiso(MPI_Comm comm, int n, int mtype = 11);

    ~SparseMatrixSolvePardiso() = default;

    // Non-copyable, non-movable (wraps PardisoMPI which is non-copyable).
    SparseMatrixSolvePardiso(const SparseMatrixSolvePardiso&) = delete;
    SparseMatrixSolvePardiso& operator=(const SparseMatrixSolvePardiso&) = delete;
    SparseMatrixSolvePardiso(SparseMatrixSolvePardiso&&) = delete;
    SparseMatrixSolvePardiso& operator=(SparseMatrixSolvePardiso&&) = delete;

    /// Set (or replace) the sparse matrix from COO triplets and re-factorize.
    ///
    /// Collective — every rank must call this with the **same** triplets.
    /// Triplets with duplicate (row, col) pairs are summed.
    ///
    /// @param triplets  Triplets with 0-based row/col indices.
    void update(std::vector<Triplet> const& triplets);

    /// Solve A x = rhs.
    ///
    /// Collective — every rank must call this with the **same** global
    /// RHS vector of size n.  On return, every rank receives the full
    /// global solution.
    ///
    /// @param rhs  Global RHS vector of size n (may be used as scratch
    ///             on rank 0).
    /// @param sol  On return, the global solution vector of size n.
    void solve(double* rhs, double* sol);

private:
    int n_;
    PardisoMPI pardiso_;

    // Full CSR storage (1-based, PARDISO convention).
    std::vector<int>    ia_;
    std::vector<int>    ja_;
    std::vector<double> a_;

    /// Convert triplets to 1-based CSR, summing duplicates.
    void triplets_to_csr(std::vector<Triplet> const& triplets);
};
