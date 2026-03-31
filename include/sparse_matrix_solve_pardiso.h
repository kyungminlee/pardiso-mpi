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
/// Every rank must supply the **same** complete set of triplets and the
/// **same** complete RHS vector.  The wrapper partitions rows across MPI
/// ranks and uses distributed assembled input (`iparm[39] = 2`) so that
/// each rank only stores and factorizes its own rows.  The solution is
/// assembled on every rank via `MPI_Allgatherv`.
///
/// Usage (combined factorization):
///   SparseMatrixSolvePardiso solver(MPI_COMM_WORLD, n);
///   solver.update(triplets);              // provide matrix, factorize
///   solver.solve(rhs, sol);               // solve A*sol = rhs
///
/// Usage (separate factorization phases):
///   SparseMatrixSolvePardiso solver(MPI_COMM_WORLD, n);
///   solver.set_triplets(triplets);        // provide matrix
///   solver.symbolic_factorize();          // symbolic factorization only
///   solver.numeric_factorize();           // numeric factorization only
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

    /// Set (or replace) the sparse matrix from COO triplets without
    /// factorizing.
    ///
    /// Collective — every rank must call this with the **same** triplets.
    /// Triplets with duplicate (row, col) pairs are summed.
    ///
    /// @param triplets  Triplets with 0-based row/col indices.
    void set_triplets(std::vector<Triplet> const& triplets);

    /// Perform symbolic factorization only (phase 11).
    ///
    /// Collective — every rank must call this.
    /// Requires set_triplets() (or update()) to have been called first.
    void symbolic_factorize();

    /// Perform numeric factorization only (phase 22).
    ///
    /// Collective — every rank must call this.
    /// Requires symbolic_factorize() to have been called first.
    void numeric_factorize();

    /// Set (or replace) the sparse matrix from COO triplets and re-factorize.
    ///
    /// Collective — every rank must call this with the **same** triplets.
    /// Triplets with duplicate (row, col) pairs are summed.
    /// Equivalent to calling set_triplets() followed by
    /// symbolic_factorize() and numeric_factorize().
    ///
    /// @param triplets  Triplets with 0-based row/col indices.
    void update(std::vector<Triplet> const& triplets);

    /// Solve A x = rhs (or A^T x = rhs, or A^H x = rhs).
    ///
    /// Collective — every rank must call this with the **same** global
    /// RHS vector of size n.  On return, every rank receives the full
    /// global solution.
    ///
    /// @param rhs        Global RHS vector of size n (may be used as scratch
    ///                   on rank 0).
    /// @param sol        On return, the global solution vector of size n.
    /// @param solve_type Which system to solve (default: normal, i.e. A x = rhs).
    void solve(double* rhs, double* sol,
               PardisoMPI::SolveType solve_type = PardisoMPI::SolveType::normal);

private:
    int n_;
    int rank_;
    int comm_size_;
    int first_row_;   ///< First row owned by this rank (1-based).
    int last_row_;    ///< Last row owned by this rank (1-based).
    PardisoMPI pardiso_;

    // Local CSR storage (1-based, PARDISO convention).
    std::vector<int>    ia_;
    std::vector<int>    ja_;
    std::vector<double> a_;

    /// Convert triplets to local 1-based CSR (only rows owned by this
    /// rank), summing duplicates.
    void triplets_to_csr(std::vector<Triplet> const& triplets);
};
