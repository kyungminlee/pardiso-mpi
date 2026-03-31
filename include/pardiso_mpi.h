#pragma once

#include <mpi.h>
#include <vector>

/// @brief MPI wrapper around Intel MKL Cluster PARDISO.
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
class PardisoMPI {
public:
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
    ///  Common values:
    ///   *  1 – real structurally symmetric
    ///   *  2 – real symmetric positive definite
    ///   * -2 – real symmetric indefinite
    ///   * 11 – real unsymmetric (default)
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
                    const double* a);

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
    void solve(const double* rhs, double* sol,
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
    int   mtype_;           ///< Matrix type (default 11 = real unsymmetric).
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
    std::vector<int>    local_ia_;
    std::vector<int>    local_ja_;
    std::vector<double> local_a_;

    // ---- Gather metadata for MPI_Allgatherv in solve ----
    std::vector<int> gather_counts_;  ///< local_nrows per rank.
    std::vector<int> gather_displs_;  ///< Displacement per rank.

    /// Initialise iparm and pt arrays for a fresh factorization.
    void init_pardiso_params();

    /// Release Cluster PARDISO internal memory (phase -1).
    /// Collective — all ranks must participate.
    void pardiso_cleanup();
};
