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
/// rank provides only the rows it owns.  After a solve, the full
/// solution is assembled on every rank via `MPI_Allgatherv`.
///
/// Matrix storage follows **1-based CSR** indexing (PARDISO convention).
class PardisoMPI {
public:
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
    /// supplies only the rows it owns.  Row ranges across all ranks must
    /// be non-overlapping and together cover rows [1, n].
    ///
    /// @param n          Global matrix dimension (rows = cols).
    /// @param first_row  First row owned by this rank (1-based, inclusive).
    /// @param last_row   Last row owned by this rank (1-based, inclusive).
    ///                   Pass first_row > last_row if this rank owns no rows.
    /// @param ia  Local row pointer array of size `local_nrows + 1` (1-based).
    /// @param ja  Local column index array of size `ia[local_nrows] - 1`
    ///            (global 1-based column indices).
    /// @param a   Local value array, same length as `ja`.
    void set_matrix(int n, int first_row, int last_row,
                    const int* ia,
                    const int* ja,
                    const double* a);

    /// Perform symbolic (phase 11) and numeric (phase 22) factorization.
    ///
    /// Collective — every rank must call this.
    void factorize();

    /// Solve A x = rhs.
    ///
    /// Collective — every rank must call this.
    /// Every rank must supply the full global RHS of size n.  On return,
    /// every rank holds the full global solution of size n (via
    /// MPI_Allgatherv).
    ///
    /// @param rhs  Global RHS vector of size n.
    /// @param sol  On return, the global solution of size n on every rank.
    void solve(const double* rhs, double* sol);

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
    bool  factorized_;      ///< True after successful factorize().
    bool  matrix_set_;      ///< True after successful set_matrix().

    // ---- Global matrix dimension ----
    int n_;                 ///< Global number of rows/columns.

    // ---- Row ownership (1-based, inclusive) ----
    int first_row_;         ///< First row owned by this rank.
    int last_row_;          ///< Last row owned by this rank.

    // ---- Local CSR (rows owned by this rank) ----
    std::vector<int>    local_ia_;
    std::vector<int>    local_ja_;
    std::vector<double> local_a_;

    // ---- Gather metadata for MPI_Allgatherv in solve ----
    std::vector<int> gather_counts_;  ///< local_nrows per rank.
    std::vector<int> gather_displs_;  ///< Displacement per rank.

    /// Release Cluster PARDISO internal memory (phase -1).
    /// Collective — all ranks must participate.
    void pardiso_cleanup();
};
