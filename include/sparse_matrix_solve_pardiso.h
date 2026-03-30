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
/// Usage:
///   SparseMatrixSolvePardiso solver(MPI_COMM_WORLD, n);
///   solver.update(triplets, rowToRank);   // provide matrix + ownership
///   solver.solve(rhs, sol);               // solve A*sol = rhs
///
/// The `rhs` pointer passed to solve() points to the *global* RHS vector
/// (size n) and may be used as scratch space.  `sol` receives the global
/// solution (size n).  Both arrays must be allocated by the caller but the
/// class manages all internal communication buffers.
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
    /// @param triplets   Triplets with 0-based row/col indices.  Duplicates
    ///                   for the same (row, col) are summed.  Each rank may
    ///                   provide the full set of triplets or only those for
    ///                   its own rows — non-local rows are ignored.
    /// @param rowToRank  Array of size n.  `rowToRank[i]` is the MPI rank
    ///                   that owns global row `i`.
    void update(std::vector<Triplet> const& triplets,
                std::vector<int> const& rowToRank);

    /// Solve A x = rhs.
    ///
    /// @param rhs  Global RHS vector of size n.  Each rank must supply all n
    ///             entries (or at minimum the entries for its local rows).
    ///             May be used as a scratch buffer and modified on return.
    /// @param sol  On return, the global solution vector of size n.
    ///             All ranks receive the full solution.
    void solve(double* rhs, double* sol);

private:
    int n_;
    MPI_Comm comm_;
    int rank_;
    int comm_size_;

    PardisoMPI pardiso_;

    // Owned CSR storage (1-based, PARDISO convention).
    std::vector<int>    ia_;
    std::vector<int>    ja_;
    std::vector<double> a_;

    // Row ownership.
    std::vector<int> row_to_rank_;
    std::vector<int> local_rows_;

    // Buffers reused across solve() calls.
    std::vector<double> local_rhs_;
    std::vector<double> local_sol_;

    /// Convert triplets to 1-based CSR, summing duplicates.
    void triplets_to_csr(std::vector<Triplet> const& triplets);
};
