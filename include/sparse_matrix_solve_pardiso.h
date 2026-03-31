#pragma once

#include "pardiso_mpi.h"
#include <mpi.h>
#include <vector>

#include <algorithm>
#include <stdexcept>

/// A sparse-matrix triplet (row, col, value) using 0-based indices.
template <typename Scalar>
struct Triplet {
    int row;
    int col;
    Scalar value;
};

/// High-level MPI-parallel sparse solver wrapper around PardisoMPI.
///
/// @tparam Scalar  Value type — `double` or `std::complex<double>`.
///
/// Every rank must supply the **same** complete set of triplets and the
/// **same** complete RHS vector.  The wrapper partitions rows across MPI
/// ranks and uses distributed assembled input (`iparm[39] = 2`) so that
/// each rank only stores and factorizes its own rows.  The solution is
/// assembled on every rank via `MPI_Allgatherv`.
///
/// Usage (combined factorization):
///   SparseMatrixSolvePardiso<double> solver(MPI_COMM_WORLD, n);
///   solver.update(triplets);              // provide matrix, factorize
///   solver.solve(rhs, sol);               // solve A*sol = rhs
///
/// Usage (separate factorization phases):
///   SparseMatrixSolvePardiso<double> solver(MPI_COMM_WORLD, n);
///   solver.set_triplets(triplets);        // provide matrix
///   solver.symbolic_factorize();          // symbolic factorization only
///   solver.numeric_factorize();           // numeric factorization only
///   solver.solve(rhs, sol);               // solve A*sol = rhs
template <typename Scalar>
class SparseMatrixSolvePardiso {
public:
    using solve_type = typename PardisoMPI<Scalar>::SolveType;

    /// @param comm   MPI communicator.
    /// @param n      Global matrix dimension (rows = cols).
    /// @param mtype  PARDISO matrix type (default depends on Scalar).
    SparseMatrixSolvePardiso(MPI_Comm comm, int n,
                             int mtype = PardisoMPI<Scalar>::default_matrix_type);

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
    void set_triplets(std::vector<Triplet<Scalar>> const& triplets);

    /// Perform symbolic factorization only (phase 11).
    void symbolic_factorize();

    /// Perform numeric factorization only (phase 22).
    void numeric_factorize();

    /// Set (or replace) the sparse matrix from COO triplets and re-factorize.
    ///
    /// Collective — every rank must call this with the **same** triplets.
    /// Equivalent to set_triplets() + symbolic_factorize() + numeric_factorize().
    void update(std::vector<Triplet<Scalar>> const& triplets);

    /// Solve A x = rhs (or A^T x = rhs, or A^H x = rhs).
    ///
    /// Collective — every rank must call this with the **same** global
    /// RHS vector of size n.  On return, every rank receives the full
    /// global solution.
    void solve(Scalar* rhs, Scalar* sol,
               solve_type st = solve_type::normal);

private:
    int n_;
    int rank_;
    int comm_size_;
    int first_row_;   ///< First row owned by this rank (1-based).
    int last_row_;    ///< Last row owned by this rank (1-based).
    PardisoMPI<Scalar> pardiso_;

    // Local CSR storage (1-based, PARDISO convention).
    std::vector<int>      ia_;
    std::vector<int>      ja_;
    std::vector<Scalar>   a_;

    /// Convert triplets to local 1-based CSR (only rows owned by this
    /// rank), summing duplicates.
    void triplets_to_csr(std::vector<Triplet<Scalar>> const& triplets);
};

// ===========================================================================
// Implementation
// ===========================================================================

template <typename Scalar>
SparseMatrixSolvePardiso<Scalar>::SparseMatrixSolvePardiso(
    MPI_Comm comm, int n, int mtype)
    : n_(n)
    , pardiso_(comm)
{
    MPI_Comm_rank(comm, &rank_);
    MPI_Comm_size(comm, &comm_size_);

    int rows_per_rank = n_ / comm_size_;
    int remainder     = n_ % comm_size_;
    int first_row_0   = rank_ * rows_per_rank + std::min(rank_, remainder);
    int local_nrows   = rows_per_rank + (rank_ < remainder ? 1 : 0);
    first_row_ = first_row_0 + 1;
    last_row_  = first_row_0 + local_nrows;

    pardiso_.set_matrix_type(mtype);
}

template <typename Scalar>
void SparseMatrixSolvePardiso<Scalar>::set_triplets(
    std::vector<Triplet<Scalar>> const& triplets)
{
    triplets_to_csr(triplets);

    int local_nrows = last_row_ - first_row_ + 1;
    std::vector<int> owned_rows(local_nrows);
    for (int i = 0; i < local_nrows; ++i)
        owned_rows[i] = first_row_ + i;

    pardiso_.set_matrix(n_, local_nrows, owned_rows.data(),
                        ia_.data(), ja_.data(), a_.data());
}

template <typename Scalar>
void SparseMatrixSolvePardiso<Scalar>::symbolic_factorize()
{
    pardiso_.symbolic_factorize();
}

template <typename Scalar>
void SparseMatrixSolvePardiso<Scalar>::numeric_factorize()
{
    pardiso_.numeric_factorize();
}

template <typename Scalar>
void SparseMatrixSolvePardiso<Scalar>::update(
    std::vector<Triplet<Scalar>> const& triplets)
{
    set_triplets(triplets);
    pardiso_.factorize();
}

template <typename Scalar>
void SparseMatrixSolvePardiso<Scalar>::solve(
    Scalar* rhs, Scalar* sol, solve_type st)
{
    pardiso_.solve(rhs, sol, st);
}

template <typename Scalar>
void SparseMatrixSolvePardiso<Scalar>::triplets_to_csr(
    std::vector<Triplet<Scalar>> const& triplets)
{
    const int first_row_0 = first_row_ - 1;
    const int local_nrows = last_row_ - first_row_ + 1;

    struct Entry {
        int row;
        int col;
        Scalar val;
    };

    std::vector<Entry> entries;
    entries.reserve(triplets.size());
    for (auto const& t : triplets) {
        if (t.row < 0 || t.row >= n_ || t.col < 0 || t.col >= n_)
            throw std::runtime_error(
                "SparseMatrixSolvePardiso: triplet index out of range");
        if (t.row >= first_row_0 && t.row < first_row_0 + local_nrows)
            entries.push_back({t.row - first_row_0, t.col, t.value});
    }

    std::sort(entries.begin(), entries.end(),
              [](Entry const& a, Entry const& b) {
                  return (a.row != b.row) ? (a.row < b.row) : (a.col < b.col);
              });

    ia_.assign(local_nrows + 1, 0);
    ja_.clear();
    a_.clear();

    ja_.reserve(entries.size());
    a_.reserve(entries.size());

    int prev_row = -1;
    int prev_col = -1;

    for (auto const& e : entries) {
        if (e.row == prev_row && e.col == prev_col) {
            a_.back() += e.val;
        } else {
            ja_.push_back(e.col + 1);
            a_.push_back(e.val);
            ia_[e.row + 1]++;
            prev_row = e.row;
            prev_col = e.col;
        }
    }

    ia_[0] = 1;
    for (int i = 1; i <= local_nrows; ++i) {
        ia_[i] += ia_[i - 1];
    }
}
