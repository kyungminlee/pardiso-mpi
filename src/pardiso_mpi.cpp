#include "pardiso_mpi.h"

#include <mkl_cluster_sparse_solver.h>
#include <cstring>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// Small helper: throw on non-zero PARDISO error
// ---------------------------------------------------------------------------
static void check_pardiso_error(int error, const char* phase_name)
{
    if (error != 0) {
        throw std::runtime_error(
            std::string("PARDISO error during ") + phase_name +
            ": error code " + std::to_string(error));
    }
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
PardisoMPI::PardisoMPI(MPI_Comm comm)
    : mtype_(11)
    , maxfct_(1)
    , mnum_(1)
    , msglvl_(0)
    , factorized_(false)
    , matrix_set_(false)
    , n_(0)
    , first_row_(0)
    , last_row_(-1)
{
    // Duplicate communicator so the caller can safely free theirs.
    int mpi_err = MPI_Comm_dup(comm, &comm_);
    if (mpi_err != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_dup failed");

    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &comm_size_);

    // Cluster PARDISO takes a Fortran communicator handle.
    comm_fortran_ = MPI_Comm_c2f(comm_);

    // Zero-initialise PARDISO internal pointers and iparm.
    std::memset(pt_,    0, sizeof(pt_));
    std::memset(iparm_, 0, sizeof(iparm_));
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------
PardisoMPI::~PardisoMPI()
{
    pardiso_cleanup();
    MPI_Comm_free(&comm_);
}

// ---------------------------------------------------------------------------
// set_matrix_type
// ---------------------------------------------------------------------------
void PardisoMPI::set_matrix_type(int mtype)
{
    mtype_ = mtype;
}

// ---------------------------------------------------------------------------
// set_matrix — each rank stores its local rows
// ---------------------------------------------------------------------------
void PardisoMPI::set_matrix(int n, int first_row, int last_row,
                            const int* ia,
                            const int* ja,
                            const double* a)
{
    // If a previous factorization exists, clean it up first.
    if (factorized_) {
        pardiso_cleanup();
        factorized_ = false;
    }

    n_ = n;
    first_row_ = first_row;
    last_row_  = last_row;

    const int local_nrows = last_row - first_row + 1;

    if (local_nrows > 0) {
        const int local_nnz = ia[local_nrows] - 1; // ia is 1-based
        local_ia_.assign(ia, ia + local_nrows + 1);
        local_ja_.assign(ja, ja + local_nnz);
        local_a_.assign(a, a + local_nnz);
    } else {
        local_ia_.clear();
        local_ja_.clear();
        local_a_.clear();
    }

    // Precompute gather metadata for MPI_Allgatherv in solve().
    gather_counts_.resize(comm_size_);
    gather_displs_.resize(comm_size_);
    MPI_Allgather(&local_nrows, 1, MPI_INT,
                  gather_counts_.data(), 1, MPI_INT, comm_);
    gather_displs_[0] = 0;
    for (int i = 1; i < comm_size_; ++i)
        gather_displs_[i] = gather_displs_[i - 1] + gather_counts_[i - 1];

    matrix_set_ = true;
}

// ---------------------------------------------------------------------------
// factorize  (collective: phases 11 + 22)
// ---------------------------------------------------------------------------
void PardisoMPI::factorize()
{
    if (!matrix_set_)
        throw std::runtime_error("factorize() called before set_matrix()");

    // (Re-)initialise PARDISO state on every rank.
    std::memset(pt_,    0, sizeof(pt_));
    std::memset(iparm_, 0, sizeof(iparm_));

    // Set parameters.
    iparm_[0]  = 1;   // Use custom values (not defaults).
    iparm_[1]  = 2;   // Nested dissection from METIS.
    iparm_[34] = 0;   // 1-based indexing.
    iparm_[39] = 2;   // Distributed assembled input.
    iparm_[40] = first_row_;   // First row owned by this rank (1-based).
    iparm_[41] = last_row_;    // Last row owned by this rank (1-based).

    int phase, error;
    int nrhs = 0;   // No RHS during factorization.
    int idum = 0;
    double ddum = 0.0;

    // Each rank passes its local data (or dummy pointers if no rows).
    double* a_ptr  = !local_a_.empty()  ? local_a_.data()  : &ddum;
    int*    ia_ptr = !local_ia_.empty() ? local_ia_.data() : &idum;
    int*    ja_ptr = !local_ja_.empty() ? local_ja_.data() : &idum;

    // Phase 11: symbolic factorization (collective).
    phase = 11;
    cluster_sparse_solver(pt_, &maxfct_, &mnum_, &mtype_, &phase,
                          &n_, a_ptr, ia_ptr, ja_ptr,
                          &idum, &nrhs, iparm_, &msglvl_,
                          &ddum, &ddum, &comm_fortran_, &error);
    check_pardiso_error(error, "symbolic factorization (phase 11)");

    // Phase 22: numerical factorization (collective).
    phase = 22;
    cluster_sparse_solver(pt_, &maxfct_, &mnum_, &mtype_, &phase,
                          &n_, a_ptr, ia_ptr, ja_ptr,
                          &idum, &nrhs, iparm_, &msglvl_,
                          &ddum, &ddum, &comm_fortran_, &error);
    check_pardiso_error(error, "numerical factorization (phase 22)");

    factorized_ = true;
}

// ---------------------------------------------------------------------------
// solve  (collective: phase 33, then allgather)
// ---------------------------------------------------------------------------
void PardisoMPI::solve(const double* rhs, double* sol)
{
    if (!factorized_)
        throw std::runtime_error("solve() called before factorize()");

    int phase = 33;
    int nrhs  = 1;
    int error = 0;
    int idum  = 0;
    double ddum = 0.0;

    double* a_ptr  = !local_a_.empty()  ? local_a_.data()  : &ddum;
    int*    ia_ptr = !local_ia_.empty() ? local_ia_.data() : &idum;
    int*    ja_ptr = !local_ja_.empty() ? local_ja_.data() : &idum;

    const int local_nrows = last_row_ - first_row_ + 1;

    if (local_nrows > 0) {
        // Extract this rank's portion of the RHS.  Cluster PARDISO may
        // modify the buffer, so work on a copy.
        std::vector<double> local_rhs(rhs + (first_row_ - 1),
                                      rhs + last_row_);
        std::vector<double> local_sol(local_nrows);

        cluster_sparse_solver(pt_, &maxfct_, &mnum_, &mtype_, &phase,
                              &n_, a_ptr, ia_ptr, ja_ptr,
                              &idum, &nrhs, iparm_, &msglvl_,
                              local_rhs.data(), local_sol.data(),
                              &comm_fortran_, &error);
        check_pardiso_error(error, "solve (phase 33)");

        // Assemble the full solution on every rank.
        MPI_Allgatherv(local_sol.data(), local_nrows, MPI_DOUBLE,
                       sol, gather_counts_.data(), gather_displs_.data(),
                       MPI_DOUBLE, comm_);
    } else {
        // This rank owns no rows — still must participate collectively.
        cluster_sparse_solver(pt_, &maxfct_, &mnum_, &mtype_, &phase,
                              &n_, a_ptr, ia_ptr, ja_ptr,
                              &idum, &nrhs, iparm_, &msglvl_,
                              &ddum, &ddum, &comm_fortran_, &error);
        check_pardiso_error(error, "solve (phase 33)");

        MPI_Allgatherv(nullptr, 0, MPI_DOUBLE,
                       sol, gather_counts_.data(), gather_displs_.data(),
                       MPI_DOUBLE, comm_);
    }
}

// ---------------------------------------------------------------------------
// pardiso_cleanup — release internal memory (collective, phase -1)
// ---------------------------------------------------------------------------
void PardisoMPI::pardiso_cleanup()
{
    if (factorized_ || matrix_set_) {
        int phase = -1;
        int nrhs  = 0;
        int error = 0;
        int idum  = 0;
        double ddum = 0.0;

        // Ensure distributed-input parameters are set for cleanup.
        iparm_[39] = 2;
        iparm_[40] = first_row_;
        iparm_[41] = last_row_;

        double* a_ptr  = !local_a_.empty()  ? local_a_.data()  : &ddum;
        int*    ia_ptr = !local_ia_.empty() ? local_ia_.data() : &idum;
        int*    ja_ptr = !local_ja_.empty() ? local_ja_.data() : &idum;

        cluster_sparse_solver(pt_, &maxfct_, &mnum_, &mtype_, &phase,
                              &n_, a_ptr, ia_ptr, ja_ptr,
                              &idum, &nrhs, iparm_, &msglvl_,
                              &ddum, &ddum, &comm_fortran_, &error);
        // Ignore errors during cleanup — we may be in a destructor path.
    }

    // Clear stored data.
    local_ia_.clear();
    local_ja_.clear();
    local_a_.clear();

    // Re-zero PARDISO handles.
    std::memset(pt_, 0, sizeof(pt_));

    factorized_ = false;
    matrix_set_ = false;
}
