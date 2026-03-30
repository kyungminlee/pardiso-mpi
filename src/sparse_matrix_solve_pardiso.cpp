#include "sparse_matrix_solve_pardiso.h"

#include <algorithm>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
SparseMatrixSolvePardiso::SparseMatrixSolvePardiso(MPI_Comm comm, int n,
                                                   int mtype)
    : n_(n)
    , pardiso_(comm)
{
    pardiso_.set_matrix_type(mtype);
}

// ---------------------------------------------------------------------------
// update — convert triplets to CSR, set matrix, factorize
// ---------------------------------------------------------------------------
void SparseMatrixSolvePardiso::update(std::vector<Triplet> const& triplets,
                                      std::vector<int> const& /*rowToRank*/)
{
    // Convert COO triplets to 1-based CSR.
    triplets_to_csr(triplets);

    // Every rank has the full CSR; pass it directly to rank 0 for
    // factorization — no per-rank distribution needed.
    pardiso_.set_global_matrix(n_, ia_.data(), ja_.data(), a_.data());
    pardiso_.factorize();
}

// ---------------------------------------------------------------------------
// solve — rank 0 solves, broadcast result to all ranks
// ---------------------------------------------------------------------------
void SparseMatrixSolvePardiso::solve(double* rhs, double* sol)
{
    // Every rank has the full global RHS.  Rank 0 solves and the
    // result is broadcast to all ranks — no gather/scatter needed.
    pardiso_.solve_global(rhs, sol);
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
