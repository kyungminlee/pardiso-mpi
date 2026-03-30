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
// set_matrix — store full CSR on rank 0
// ---------------------------------------------------------------------------
void PardisoMPI::set_matrix(int n,
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

    // With iparm[39]=0 (centralised input), only rank 0 needs the matrix.
    if (rank_ == 0) {
        const int nnz = ia[n] - 1;   // ia is 1-based: nnz = ia[n] - ia[0]
        global_ia_.assign(ia, ia + n + 1);
        global_ja_.assign(ja, ja + nnz);
        global_a_.assign(a, a + nnz);
    }

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
    iparm_[39] = 0;   // Centralised input: matrix on rank 0 only.

    int phase, error;
    int nrhs = 0;   // No RHS during factorization.
    int idum = 0;
    double ddum = 0.0;

    // Rank 0 passes real data; other ranks pass dummy pointers.
    double* a_ptr  = (rank_ == 0) ? global_a_.data()  : &ddum;
    int*    ia_ptr = (rank_ == 0) ? global_ia_.data() : &idum;
    int*    ja_ptr = (rank_ == 0) ? global_ja_.data() : &idum;

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
// solve  (collective: phase 33, then broadcast)
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

    double* a_ptr  = (rank_ == 0) ? global_a_.data()  : &ddum;
    int*    ia_ptr = (rank_ == 0) ? global_ia_.data() : &idum;
    int*    ja_ptr = (rank_ == 0) ? global_ja_.data() : &idum;

    if (rank_ == 0) {
        // Cluster PARDISO may modify the RHS buffer, so work on a copy.
        std::vector<double> rhs_copy(rhs, rhs + n_);

        cluster_sparse_solver(pt_, &maxfct_, &mnum_, &mtype_, &phase,
                              &n_, a_ptr, ia_ptr, ja_ptr,
                              &idum, &nrhs, iparm_, &msglvl_,
                              rhs_copy.data(), sol, &comm_fortran_, &error);
    } else {
        cluster_sparse_solver(pt_, &maxfct_, &mnum_, &mtype_, &phase,
                              &n_, a_ptr, ia_ptr, ja_ptr,
                              &idum, &nrhs, iparm_, &msglvl_,
                              &ddum, &ddum, &comm_fortran_, &error);
    }
    check_pardiso_error(error, "solve (phase 33)");

    // Broadcast the full solution to all ranks.
    MPI_Bcast(sol, n_, MPI_DOUBLE, 0, comm_);
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

        double* a_ptr  = (rank_ == 0 && !global_a_.empty())  ? global_a_.data()  : &ddum;
        int*    ia_ptr = (rank_ == 0 && !global_ia_.empty()) ? global_ia_.data() : &idum;
        int*    ja_ptr = (rank_ == 0 && !global_ja_.empty()) ? global_ja_.data() : &idum;

        cluster_sparse_solver(pt_, &maxfct_, &mnum_, &mtype_, &phase,
                              &n_, a_ptr, ia_ptr, ja_ptr,
                              &idum, &nrhs, iparm_, &msglvl_,
                              &ddum, &ddum, &comm_fortran_, &error);
        // Ignore errors during cleanup — we may be in a destructor path.
    }

    // Clear stored data.
    global_ia_.clear();
    global_ja_.clear();
    global_a_.clear();

    // Re-zero PARDISO handles.
    std::memset(pt_, 0, sizeof(pt_));

    factorized_ = false;
    matrix_set_ = false;
}
