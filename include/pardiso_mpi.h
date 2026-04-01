#pragma once

#include <complex>
#include <mpi.h>
#include <type_traits>
#include <vector>

#include <mkl_cluster_sparse_solver.h>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace pardiso_detail {

inline void check_pardiso_error(int error, const char* phase_name)
{
    if (error != 0) {
        throw std::runtime_error(
            std::string("PARDISO error during ") + phase_name +
            ": error code " + std::to_string(error));
    }
}

template <typename T> inline MPI_Datatype mpi_scalar_type();
template <> inline MPI_Datatype mpi_scalar_type<double>()
{ return MPI_DOUBLE; }
template <> inline MPI_Datatype mpi_scalar_type<std::complex<double>>()
{ return MPI_C_DOUBLE_COMPLEX; }

} // namespace pardiso_detail

/// @brief MPI wrapper around Intel MKL Cluster PARDISO.
///
/// @tparam Scalar  Value type — `double` or `std::complex<double>`.
///
/// All phases (factorization and solve) are **collective** — every rank
/// in the communicator must call them.  The solver distributes work
/// internally across MPI ranks.
///
/// Uses **distributed assembled input** (`iparm[39] = 2`): each MPI
/// rank provides only the rows it owns.  Rows need not be contiguous —
/// the wrapper builds an internal permutation to map each rank's rows
/// to a contiguous block, remaps column indices accordingly, and
/// inverse-permutes the solution after the solve.  After a solve, the
/// full solution is assembled on every rank via `MPI_Allgatherv`.
///
/// Matrix storage follows **1-based CSR** indexing (PARDISO convention).
template <typename Scalar>
class PardisoMPI {
    static_assert(std::is_same_v<Scalar, double> ||
                  std::is_same_v<Scalar, std::complex<double>>,
                  "PardisoMPI supports double or std::complex<double>");

public:
    /// Default PARDISO matrix type for this scalar type.
    ///   11 = real unsymmetric, 13 = complex unsymmetric.
    static constexpr int default_matrix_type =
        std::is_same_v<Scalar, double> ? 11 : 13;

    /// Solver phase — tracks how far the factorization has progressed.
    enum class Phase {
        initial,       ///< No matrix set yet.
        matrix_set,    ///< Matrix data stored; no factorization performed.
        symbolic,      ///< Symbolic factorization complete (phase 11).
        numeric        ///< Numeric factorization complete (phase 22); ready to solve.
    };

    /// Controls which system is solved during the solve phase.
    enum class SolveType {
        normal,        ///< Solve A   x = rhs.
        transpose,     ///< Solve A^T x = rhs.
        adjoint        ///< Solve A^H x = rhs (same as transpose for real matrices).
    };

    /// Construct a solver bound to the given MPI communicator.
    /// The communicator is duplicated internally so the caller may
    /// free the original at any time after construction.
    explicit PardisoMPI(MPI_Comm comm);

    /// Destroy internal PARDISO data structures and free the
    /// duplicated communicator.
    ~PardisoMPI();

    // Non-copyable, non-movable (PARDISO internal state is opaque).
    PardisoMPI(const PardisoMPI&) = delete;
    PardisoMPI& operator=(const PardisoMPI&) = delete;
    PardisoMPI(PardisoMPI&&) = delete;
    PardisoMPI& operator=(PardisoMPI&&) = delete;

    /// Set the PARDISO matrix type **before** calling set_matrix().
    ///
    ///  Common real values:
    ///   *  1 – real structurally symmetric
    ///   *  2 – real symmetric positive definite
    ///   * -2 – real symmetric indefinite
    ///   * 11 – real unsymmetric
    ///
    ///  Common complex values:
    ///   *  3 – complex structurally symmetric
    ///   *  6 – complex symmetric
    ///   * -6 – complex symmetric indefinite
    ///   * 13 – complex unsymmetric
    void set_matrix_type(int mtype);

    /// Provide this rank's portion of the sparse matrix in 1-based CSR.
    ///
    /// Every rank must call this method (it is collective).  Each rank
    /// supplies only the rows it owns.  The rows need **not** be
    /// contiguous; the wrapper internally builds a symmetric permutation
    /// so that each rank's rows form a contiguous block for Cluster
    /// PARDISO, and remaps column indices accordingly.
    ///
    /// @param n           Global matrix dimension (rows = cols).
    /// @param local_nrows Number of rows owned by this rank.
    /// @param owned_rows  Array of `local_nrows` row indices owned by this
    ///                    rank (1-based).  Across all ranks, these must be
    ///                    non-overlapping and together cover [1, n].
    /// @param ia  Local row pointer array of size `local_nrows + 1` (1-based).
    ///            Row i of `ia`/`ja`/`a` corresponds to global row
    ///            `owned_rows[i]`.
    /// @param ja  Local column index array of size `ia[local_nrows] - 1`
    ///            (global 1-based column indices).
    /// @param a   Local value array, same length as `ja`.
    void set_matrix(int n, int local_nrows, const int* owned_rows,
                    const int* ia,
                    const int* ja,
                    const Scalar* a);

    /// Perform symbolic factorization only (phase 11).
    ///
    /// Collective — every rank must call this.
    /// Requires set_matrix() to have been called first.
    void symbolic_factorize();

    /// Perform numeric factorization only (phase 22).
    ///
    /// Collective — every rank must call this.
    /// Requires symbolic_factorize() to have been called first.
    /// Can be called again after symbolic_factorize() to re-do
    /// numeric factorization with updated values (same sparsity pattern).
    void numeric_factorize();

    /// Perform symbolic (phase 11) and numeric (phase 22) factorization.
    ///
    /// Convenience method equivalent to calling symbolic_factorize()
    /// followed by numeric_factorize().
    ///
    /// Collective — every rank must call this.
    void factorize();

    /// Solve A x = rhs (or A^T x = rhs, or A^H x = rhs).
    ///
    /// Collective — every rank must call this.
    /// Every rank must supply the full global RHS of size n.  On return,
    /// every rank holds the full global solution of size n (via
    /// MPI_Allgatherv).
    ///
    /// @param rhs        Global RHS vector of size n.
    /// @param sol        On return, the global solution of size n on every rank.
    /// @param solve_type Which system to solve (default: normal, i.e. A x = rhs).
    void solve(const Scalar* rhs, Scalar* sol,
               SolveType solve_type = SolveType::normal);

    /// Return the current solver phase.
    Phase phase() const { return phase_; }

private:
    // ---- MPI state ----
    MPI_Comm  comm_;
    MPI_Fint  comm_fortran_;   ///< Fortran handle for cluster_sparse_solver.
    int rank_;
    int comm_size_;

    // ---- PARDISO state ----
    void* pt_[64];          ///< PARDISO internal pointer array.
    int   iparm_[64];       ///< PARDISO integer parameters.
    int   mtype_;           ///< Matrix type.
    int   maxfct_;          ///< Max factors kept in memory.
    int   mnum_;            ///< Which factorization to use.
    int   msglvl_;          ///< Message level (0 = no output).
    Phase phase_;           ///< Current solver phase.

    // ---- Global matrix dimension ----
    int n_;                 ///< Global number of rows/columns.

    // ---- Permutation (maps non-contiguous ownership to contiguous) ----
    int first_row_;         ///< First permuted row for this rank (1-based).
    int last_row_;          ///< Last permuted row for this rank (1-based).
    std::vector<int> perm_;   ///< perm_[new] = old (0-based).
    std::vector<int> iperm_;  ///< iperm_[old] = new (0-based).

    // ---- Local CSR (permuted, contiguous rows for PARDISO) ----
    std::vector<int>      local_ia_;
    std::vector<int>      local_ja_;
    std::vector<Scalar>   local_a_;

    // ---- Gather metadata for MPI_Allgatherv in solve ----
    std::vector<int> gather_counts_;  ///< local_nrows per rank.
    std::vector<int> gather_displs_;  ///< Displacement per rank.

    /// Initialise iparm and pt arrays for a fresh factorization.
    void init_pardiso_params();

    /// Release Cluster PARDISO internal memory (phase -1).
    /// Collective — all ranks must participate.
    void pardiso_cleanup();
};

// ===========================================================================
// Implementation
// ===========================================================================

template <typename Scalar>
PardisoMPI<Scalar>::PardisoMPI(MPI_Comm comm)
    : mtype_(default_matrix_type)
    , maxfct_(1)
    , mnum_(1)
    , msglvl_(0)
    , phase_(Phase::initial)
    , n_(0)
    , first_row_(0)
    , last_row_(-1)
{
    int mpi_err = MPI_Comm_dup(comm, &comm_);
    if (mpi_err != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_dup failed");

    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &comm_size_);

    comm_fortran_ = MPI_Comm_c2f(comm_);

    std::memset(pt_,    0, sizeof(pt_));
    std::memset(iparm_, 0, sizeof(iparm_));
}

template <typename Scalar>
PardisoMPI<Scalar>::~PardisoMPI()
{
    pardiso_cleanup();
    MPI_Comm_free(&comm_);
}

template <typename Scalar>
void PardisoMPI<Scalar>::set_matrix_type(int mtype)
{
    mtype_ = mtype;
}

template <typename Scalar>
void PardisoMPI<Scalar>::set_matrix(int n, int local_nrows,
                                    const int* owned_rows,
                                    const int* ia,
                                    const int* ja,
                                    const Scalar* a)
{
    if (phase_ >= Phase::symbolic) {
        pardiso_cleanup();
    }

    n_ = n;

    // 1. Allgather row counts and displacements.
    gather_counts_.resize(comm_size_);
    gather_displs_.resize(comm_size_);
    MPI_Allgather(&local_nrows, 1, MPI_INT,
                  gather_counts_.data(), 1, MPI_INT, comm_);
    gather_displs_[0] = 0;
    for (int i = 1; i < comm_size_; ++i)
        gather_displs_[i] = gather_displs_[i - 1] + gather_counts_[i - 1];

    // 2. Allgatherv owned_rows to build the global permutation.
    std::vector<int> all_owned(n);
    MPI_Allgatherv(owned_rows, local_nrows, MPI_INT,
                   all_owned.data(), gather_counts_.data(),
                   gather_displs_.data(), MPI_INT, comm_);

    perm_.resize(n);
    iperm_.resize(n);
    for (int i = 0; i < n; ++i) {
        perm_[i] = all_owned[i] - 1;
        iperm_[perm_[i]] = i;
    }

    // 3. Contiguous row range for this rank in the permuted ordering.
    first_row_ = gather_displs_[rank_] + 1;
    last_row_  = gather_displs_[rank_] + local_nrows;

    // 4. Copy local CSR and remap column indices via the permutation.
    if (local_nrows > 0) {
        const int local_nnz = ia[local_nrows] - 1;
        local_ia_.assign(ia, ia + local_nrows + 1);
        local_ja_.resize(local_nnz);
        local_a_.resize(local_nnz);

        for (int i = 0; i < local_nrows; ++i) {
            const int start = ia[i] - 1;
            const int end   = ia[i + 1] - 1;
            const int row_nnz = end - start;

            struct ColVal { int col; Scalar val; };
            std::vector<ColVal> entries(row_nnz);
            for (int k = 0; k < row_nnz; ++k) {
                int old_col_0 = ja[start + k] - 1;
                entries[k] = {iperm_[old_col_0] + 1, a[start + k]};
            }

            std::sort(entries.begin(), entries.end(),
                      [](const ColVal& x, const ColVal& y) {
                          return x.col < y.col;
                      });

            for (int k = 0; k < row_nnz; ++k) {
                local_ja_[start + k] = entries[k].col;
                local_a_[start + k]  = entries[k].val;
            }
        }
    } else {
        local_ia_.clear();
        local_ja_.clear();
        local_a_.clear();
    }

    phase_ = Phase::matrix_set;
}

template <typename Scalar>
void PardisoMPI<Scalar>::init_pardiso_params()
{
    std::memset(pt_,    0, sizeof(pt_));
    std::memset(iparm_, 0, sizeof(iparm_));

    iparm_[0]  = 1;
    iparm_[1]  = 2;
    iparm_[34] = 0;
    iparm_[39] = 2;
    iparm_[40] = first_row_;
    iparm_[41] = last_row_;
}

template <typename Scalar>
void PardisoMPI<Scalar>::symbolic_factorize()
{
    if (phase_ < Phase::matrix_set)
        throw std::runtime_error(
            "symbolic_factorize() called before set_matrix()");

    if (phase_ >= Phase::symbolic) {
        pardiso_cleanup();
        phase_ = Phase::matrix_set;
    }

    init_pardiso_params();

    int phase = 11;
    int error = 0;
    int nrhs  = 0;
    int idum  = 0;
    Scalar sdum{};

    void* a_ptr  = !local_a_.empty()  ? static_cast<void*>(local_a_.data())  : static_cast<void*>(&sdum);
    int*  ia_ptr = !local_ia_.empty() ? local_ia_.data() : &idum;
    int*  ja_ptr = !local_ja_.empty() ? local_ja_.data() : &idum;

    cluster_sparse_solver(pt_, &maxfct_, &mnum_, &mtype_, &phase,
                          &n_, a_ptr, ia_ptr, ja_ptr,
                          &idum, &nrhs, iparm_, &msglvl_,
                          &sdum, &sdum, &comm_fortran_, &error);
    pardiso_detail::check_pardiso_error(error, "symbolic factorization (phase 11)");

    phase_ = Phase::symbolic;
}

template <typename Scalar>
void PardisoMPI<Scalar>::numeric_factorize()
{
    if (phase_ < Phase::symbolic)
        throw std::runtime_error(
            "numeric_factorize() called before symbolic_factorize()");

    int phase = 22;
    int error = 0;
    int nrhs  = 0;
    int idum  = 0;
    Scalar sdum{};

    void* a_ptr  = !local_a_.empty()  ? static_cast<void*>(local_a_.data())  : static_cast<void*>(&sdum);
    int*  ia_ptr = !local_ia_.empty() ? local_ia_.data() : &idum;
    int*  ja_ptr = !local_ja_.empty() ? local_ja_.data() : &idum;

    cluster_sparse_solver(pt_, &maxfct_, &mnum_, &mtype_, &phase,
                          &n_, a_ptr, ia_ptr, ja_ptr,
                          &idum, &nrhs, iparm_, &msglvl_,
                          &sdum, &sdum, &comm_fortran_, &error);
    pardiso_detail::check_pardiso_error(error, "numerical factorization (phase 22)");

    phase_ = Phase::numeric;
}

template <typename Scalar>
void PardisoMPI<Scalar>::factorize()
{
    symbolic_factorize();
    numeric_factorize();
}

template <typename Scalar>
void PardisoMPI<Scalar>::solve(const Scalar* rhs, Scalar* sol,
                               SolveType solve_type)
{
    if (phase_ < Phase::numeric)
        throw std::runtime_error("solve() called before numeric factorization");

    switch (solve_type) {
    case SolveType::normal:    iparm_[11] = 0; break;
    case SolveType::transpose: iparm_[11] = 2; break;
    case SolveType::adjoint:   iparm_[11] = 1; break;
    }

    int phase = 33;
    int nrhs  = 1;
    int error = 0;
    int idum  = 0;
    Scalar sdum{};

    void* a_ptr  = !local_a_.empty()  ? static_cast<void*>(local_a_.data())  : static_cast<void*>(&sdum);
    int*  ia_ptr = !local_ia_.empty() ? local_ia_.data() : &idum;
    int*  ja_ptr = !local_ja_.empty() ? local_ja_.data() : &idum;

    const int local_nrows = last_row_ - first_row_ + 1;

    // Permute the full RHS to match the reordered matrix.
    std::vector<Scalar> perm_rhs(n_);
    for (int i = 0; i < n_; ++i)
        perm_rhs[i] = rhs[perm_[i]];

    // Extract this rank's local portion and solve.
    std::vector<Scalar> local_sol(local_nrows);
    if (local_nrows > 0) {
        std::vector<Scalar> local_rhs(perm_rhs.begin() + (first_row_ - 1),
                                      perm_rhs.begin() + last_row_);

        cluster_sparse_solver(pt_, &maxfct_, &mnum_, &mtype_, &phase,
                              &n_, a_ptr, ia_ptr, ja_ptr,
                              &idum, &nrhs, iparm_, &msglvl_,
                              static_cast<void*>(local_rhs.data()),
                              static_cast<void*>(local_sol.data()),
                              &comm_fortran_, &error);
    } else {
        cluster_sparse_solver(pt_, &maxfct_, &mnum_, &mtype_, &phase,
                              &n_, a_ptr, ia_ptr, ja_ptr,
                              &idum, &nrhs, iparm_, &msglvl_,
                              &sdum, &sdum, &comm_fortran_, &error);
    }
    pardiso_detail::check_pardiso_error(error, "solve (phase 33)");

    // Assemble full permuted solution.
    const MPI_Datatype scalar_mpi_type = pardiso_detail::mpi_scalar_type<Scalar>();
    std::vector<Scalar> perm_sol(n_);
    MPI_Allgatherv(local_nrows > 0 ? local_sol.data() : nullptr,
                   local_nrows, scalar_mpi_type,
                   perm_sol.data(), gather_counts_.data(),
                   gather_displs_.data(), scalar_mpi_type, comm_);

    // Inverse-permute back to original ordering.
    for (int i = 0; i < n_; ++i)
        sol[perm_[i]] = perm_sol[i];
}

template <typename Scalar>
void PardisoMPI<Scalar>::pardiso_cleanup()
{
    if (phase_ >= Phase::symbolic) {
        int phase = -1;
        int nrhs  = 0;
        int error = 0;
        int idum  = 0;
        Scalar sdum{};

        iparm_[39] = 2;
        iparm_[40] = first_row_;
        iparm_[41] = last_row_;

        void* a_ptr  = !local_a_.empty()  ? static_cast<void*>(local_a_.data())  : static_cast<void*>(&sdum);
        int*  ia_ptr = !local_ia_.empty() ? local_ia_.data() : &idum;
        int*  ja_ptr = !local_ja_.empty() ? local_ja_.data() : &idum;

        cluster_sparse_solver(pt_, &maxfct_, &mnum_, &mtype_, &phase,
                              &n_, a_ptr, ia_ptr, ja_ptr,
                              &idum, &nrhs, iparm_, &msglvl_,
                              &sdum, &sdum, &comm_fortran_, &error);
    }

    local_ia_.clear();
    local_ja_.clear();
    local_a_.clear();

    std::memset(pt_, 0, sizeof(pt_));

    phase_ = Phase::initial;
}
