#pragma once

#include <mpi.h>
#include <vector>

/// @brief MPI wrapper around Intel MKL Cluster PARDISO.
///
/// All phases (factorization and solve) are **collective** — every rank
/// in the communicator must call them.  The solver distributes work
/// internally across MPI ranks.
///
/// With the default `iparm[39] = 0` (centralised input), only rank 0
/// needs to supply the matrix and RHS data; other ranks may pass dummy
/// pointers.  After a solve, the solution is broadcast so that every
/// rank holds the full result.
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

    /// Provide the full global sparse matrix in 1-based CSR format.
    ///
    /// Every rank must call this method (it is collective).  Only
    /// rank 0's data is read; other ranks may pass any valid pointers.
    ///
    /// @param n   Global matrix dimension (rows = cols).
    /// @param ia  Row pointer array of size `n + 1` (1-based).
    /// @param ja  Column index array of size `ia[n] - 1` (1-based).
    /// @param a   Value array, same length as `ja`.
    void set_matrix(int n,
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
    /// Rank 0 must supply the global RHS of size n.  On return, every
    /// rank holds the full global solution of size n (via MPI_Bcast).
    ///
    /// @param rhs  Global RHS vector of size n (used on rank 0).
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

    // ---- Global CSR (rank 0 only) ----
    std::vector<int>    global_ia_;
    std::vector<int>    global_ja_;
    std::vector<double> global_a_;

    /// Release Cluster PARDISO internal memory (phase -1).
    /// Collective — all ranks must participate.
    void pardiso_cleanup();
};
