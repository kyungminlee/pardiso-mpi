#include "sparse_matrix_solve_pardiso.h"

#include <algorithm>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
SparseMatrixSolvePardiso::SparseMatrixSolvePardiso(MPI_Comm comm, int n,
                                                   int mtype)
    : n_(n)
    , comm_(comm)
    , pardiso_(comm)
{
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &comm_size_);
    pardiso_.set_matrix_type(mtype);
}

// ---------------------------------------------------------------------------
// update — convert triplets to CSR, set matrix, factorize
// ---------------------------------------------------------------------------
void SparseMatrixSolvePardiso::update(std::vector<Triplet> const& triplets,
                                      std::vector<int> const& rowToRank)
{
    if (static_cast<int>(rowToRank.size()) != n_)
        throw std::runtime_error(
            "SparseMatrixSolvePardiso::update: rowToRank.size() != n");

    row_to_rank_ = rowToRank;

    // Determine local rows.
    local_rows_.clear();
    for (int i = 0; i < n_; ++i) {
        if (row_to_rank_[i] == rank_)
            local_rows_.push_back(i);
    }

    // Resize local buffers.
    local_rhs_.resize(local_rows_.size());
    local_sol_.resize(local_rows_.size());

    // Convert COO triplets to 1-based CSR.
    triplets_to_csr(triplets);

    // Feed into PardisoMPI and factorize.
    pardiso_.set_matrix(n_, ia_.data(), ja_.data(), a_.data(),
                        row_to_rank_.data());
    pardiso_.factorize();
}

// ---------------------------------------------------------------------------
// solve — extract local RHS, call PardisoMPI, scatter back to global sol
// ---------------------------------------------------------------------------
void SparseMatrixSolvePardiso::solve(double* rhs, double* sol)
{
    const int local_n = static_cast<int>(local_rows_.size());

    // Extract local portion of RHS.
    for (int li = 0; li < local_n; ++li) {
        local_rhs_[li] = rhs[local_rows_[li]];
    }

    // Solve (local RHS in, local solution out).
    pardiso_.solve(local_rhs_.data(), local_sol_.data());

    // Gather the full solution on all ranks via Allgatherv.
    // First collect counts from every rank.
    std::vector<int> counts(comm_size_);
    std::vector<int> displs(comm_size_);
    MPI_Allgather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, comm_);

    displs[0] = 0;
    for (int r = 1; r < comm_size_; ++r)
        displs[r] = displs[r - 1] + counts[r - 1];

    // Gather local row indices so every rank can reconstruct the mapping.
    int total = displs.back() + counts.back();
    std::vector<int> all_rows(total);
    MPI_Allgatherv(local_rows_.data(), local_n, MPI_INT,
                   all_rows.data(), counts.data(), displs.data(),
                   MPI_INT, comm_);

    // Gather local solution values.
    std::vector<double> all_vals(total);
    MPI_Allgatherv(local_sol_.data(), local_n, MPI_DOUBLE,
                   all_vals.data(), counts.data(), displs.data(),
                   MPI_DOUBLE, comm_);

    // Scatter into the global solution vector.
    for (int k = 0; k < total; ++k) {
        sol[all_rows[k]] = all_vals[k];
    }
}

// ---------------------------------------------------------------------------
// triplets_to_csr — COO (0-based) to CSR (1-based), summing duplicates
// ---------------------------------------------------------------------------
void SparseMatrixSolvePardiso::triplets_to_csr(
    std::vector<Triplet> const& triplets)
{
    // 1. Sort triplets by (row, col).
    struct Entry {
        int row;
        int col;
        double val;
    };

    std::vector<Entry> entries;
    entries.reserve(triplets.size());
    for (auto const& t : triplets) {
        if (t.row < 0 || t.row >= n_ || t.col < 0 || t.col >= n_)
            throw std::runtime_error(
                "SparseMatrixSolvePardiso: triplet index out of range");
        entries.push_back({t.row, t.col, t.value});
    }

    std::sort(entries.begin(), entries.end(),
              [](Entry const& a, Entry const& b) {
                  return (a.row != b.row) ? (a.row < b.row) : (a.col < b.col);
              });

    // 2. Merge duplicates and build CSR.
    ia_.assign(n_ + 1, 0);
    ja_.clear();
    a_.clear();

    ja_.reserve(entries.size());
    a_.reserve(entries.size());

    int prev_row = -1;
    int prev_col = -1;

    for (auto const& e : entries) {
        if (e.row == prev_row && e.col == prev_col) {
            // Duplicate — sum into last element.
            a_.back() += e.val;
        } else {
            ja_.push_back(e.col + 1);   // convert to 1-based
            a_.push_back(e.val);
            ia_[e.row + 1]++;           // count entries per row
            prev_row = e.row;
            prev_col = e.col;
        }
    }

    // 3. Prefix sum to build row pointers (1-based).
    ia_[0] = 1;
    for (int i = 1; i <= n_; ++i) {
        ia_[i] += ia_[i - 1];
    }
}
