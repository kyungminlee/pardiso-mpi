#pragma once

#include <mpi.h>
#include <vector>
#include <cstdint>

/// @brief MPI-aware wrapper around Intel MKL PARDISO.
///
/// Each MPI rank owns a subset of rows defined by a user-supplied
/// `row_to_rank` mapping.  During factorization and solve, rank 0
/// gathers the full CSR matrix and right-hand side, calls PARDISO
/// locally, and scatters the solution back to the owning ranks.
///
/// Matrix storage follows **1-based CSR** indexing (PARDISO convention).
/// The arrays `ia`, `ja` supplied to `set_matrix` must therefore use
/// Fortran-style 1-based indices.
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

    /// Provide the global sparse matrix in 1-based CSR format together
    /// with the row-to-rank ownership map.
    ///
    /// @param n            Global matrix dimension (number of rows/columns).
    /// @param ia           Row pointer array of size `n + 1` (1-based).
    /// @param ja           Column index array of size `ia[n] - 1` (1-based).
    /// @param a            Value array, same length as `ja`.
    /// @param row_to_rank  Array of size `n`.  `row_to_rank[i]` is the MPI
    ///                     rank that owns global row `i` (0-based row index).
    ///
    /// Each rank need only supply the rows it owns; entries for non-local
    /// rows are ignored.  Alternatively every rank may pass the full global
    /// arrays — only the locally-owned rows will be stored.
    void set_matrix(int n,
                    const int* ia,
                    const int* ja,
                    const double* a,
                    const int* row_to_rank);

    /// Perform symbolic (phase 11) and numeric (phase 22) factorization.
    /// Must be called after set_matrix().
    void factorize();

    /// Solve the system A x = rhs.
    ///
    /// @param rhs       On each rank, the local portion of the right-hand
    ///                  side — i.e. entries for the rows owned by this rank,
    ///                  in the same order they were given to set_matrix().
    /// @param solution  On return, the local portion of the solution vector
    ///                  (same size and ordering as @p rhs).
    void solve(const double* rhs, double* solution);

private:
    // ---- MPI state ----
    MPI_Comm comm_;
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

    // ---- Local (this rank) CSR data ----
    std::vector<int>    local_rows_;   ///< Sorted global row indices owned by this rank.
    std::vector<int>    local_ia_;     ///< Local row pointers (1-based).
    std::vector<int>    local_ja_;     ///< Local column indices (1-based).
    std::vector<double> local_a_;      ///< Local values.

    // ---- Gathered (rank-0 only) global CSR ----
    std::vector<int>    global_ia_;
    std::vector<int>    global_ja_;
    std::vector<double> global_a_;

    // ---- Helpers ----

    /// Gather local CSR pieces onto rank 0, rebuilding full global CSR.
    void gather_matrix();

    /// Gather local RHS pieces onto rank 0, returning the full RHS
    /// (only meaningful on rank 0).
    std::vector<double> gather_rhs(const double* local_rhs);

    /// Scatter the global solution from rank 0 to each rank's local portion.
    void scatter_solution(const double* global_sol, double* local_sol);

    /// Release PARDISO internal memory (phase -1).  Safe to call even if
    /// no factorization was performed.
    void pardiso_cleanup();
};
