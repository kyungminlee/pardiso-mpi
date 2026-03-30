#include "pardiso_mpi.h"

#include <mkl_pardiso.h>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <numeric>
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
    , global_mode_(false)
    , n_(0)
{
    // Duplicate communicator so the caller can safely free theirs.
    int mpi_err = MPI_Comm_dup(comm, &comm_);
    if (mpi_err != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_dup failed");

    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &comm_size_);

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
// set_matrix
// ---------------------------------------------------------------------------
void PardisoMPI::set_matrix(int n,
                            const int* ia,
                            const int* ja,
                            const double* a,
                            const int* row_to_rank)
{
    // If a previous factorization exists, clean it up first.
    if (factorized_) {
        pardiso_cleanup();
        factorized_ = false;
    }

    n_ = n;
    global_mode_ = false;

    // ----------------------------------------------------------------
    // 1.  Determine which global rows this rank owns.
    // ----------------------------------------------------------------
    local_rows_.clear();
    for (int i = 0; i < n; ++i) {
        if (row_to_rank[i] == rank_) {
            local_rows_.push_back(i);
        }
    }
    // local_rows_ is already sorted because we iterate in order.

    // ----------------------------------------------------------------
    // 2.  Extract the local CSR sub-matrix for those rows.
    //     Input ia/ja are 1-based.  Output local_ia_/local_ja_ are
    //     also 1-based (column indices unchanged; row pointers
    //     re-based to start at 1).
    // ----------------------------------------------------------------
    const int local_n = static_cast<int>(local_rows_.size());

    local_ia_.resize(local_n + 1);
    local_ja_.clear();
    local_a_.clear();

    int running = 1;                       // 1-based row-pointer start
    for (int li = 0; li < local_n; ++li) {
        const int gi = local_rows_[li];    // global row index (0-based)
        const int row_start = ia[gi]   - 1;  // convert to 0-based offset
        const int row_end   = ia[gi+1] - 1;
        local_ia_[li] = running;
        for (int k = row_start; k < row_end; ++k) {
            local_ja_.push_back(ja[k]);    // already 1-based
            local_a_.push_back(a[k]);
        }
        running += (row_end - row_start);
    }
    local_ia_[local_n] = running;

    matrix_set_ = true;
}

// ---------------------------------------------------------------------------
// set_global_matrix — store full CSR directly on rank 0
// ---------------------------------------------------------------------------
void PardisoMPI::set_global_matrix(int n,
                                   const int* ia,
                                   const int* ja,
                                   const double* a)
{
    if (factorized_) {
        pardiso_cleanup();
        factorized_ = false;
    }

    n_ = n;
    global_mode_ = true;

    // Clear distributed-mode data (not used in global mode).
    local_rows_.clear();
    local_ia_.clear();
    local_ja_.clear();
    local_a_.clear();

    if (rank_ == 0) {
        const int nnz = ia[n] - 1;   // ia is 1-based: nnz = ia[n] - ia[0]
        global_ia_.assign(ia, ia + n + 1);
        global_ja_.assign(ja, ja + nnz);
        global_a_.assign(a, a + nnz);
    }

    matrix_set_ = true;
}

// ---------------------------------------------------------------------------
// factorize  (phases 11 + 22 on rank 0)
// ---------------------------------------------------------------------------
void PardisoMPI::factorize()
{
    if (!matrix_set_)
        throw std::runtime_error("factorize() called before set_matrix()");

    // In global mode the full CSR is already on rank 0; skip the gather.
    if (!global_mode_)
        gather_matrix();

    // ----- rank 0: initialise PARDISO and run phases 11 + 22 -----
    if (rank_ == 0) {
        // Re-initialise PARDISO state.
        std::memset(pt_,    0, sizeof(pt_));
        std::memset(iparm_, 0, sizeof(iparm_));

        // Use pardisoinit to set reasonable defaults.
        pardisoinit(pt_, &mtype_, iparm_);

        // Customise selected parameters.
        iparm_[0]  = 1;  // No default values (we set our own).
        iparm_[1]  = 2;  // Nested dissection from METIS.
        iparm_[34] = 1;  // 0-based indexing? No — keep 1-based:
        // Actually iparm[34]=1 means 0-based. We want 1-based (default=0).
        iparm_[34] = 0;  // 1-based indexing.

        int phase, error;
        int nrhs = 0;   // No RHS during factorization.
        int idum = 0;
        double ddum = 0.0;

        // Phase 11: symbolic factorization.
        phase = 11;
        pardiso(pt_, &maxfct_, &mnum_, &mtype_, &phase,
                &n_, global_a_.data(), global_ia_.data(), global_ja_.data(),
                &idum, &nrhs, iparm_, &msglvl_, &ddum, &ddum, &error);
        check_pardiso_error(error, "symbolic factorization (phase 11)");

        // Phase 22: numerical factorization.
        phase = 22;
        pardiso(pt_, &maxfct_, &mnum_, &mtype_, &phase,
                &n_, global_a_.data(), global_ia_.data(), global_ja_.data(),
                &idum, &nrhs, iparm_, &msglvl_, &ddum, &ddum, &error);
        check_pardiso_error(error, "numerical factorization (phase 22)");
    }

    // Broadcast success to all ranks (implicitly via no exception).
    // We use a barrier so all ranks stay synchronised.
    MPI_Barrier(comm_);

    factorized_ = true;
}

// ---------------------------------------------------------------------------
// solve  (phase 33 on rank 0, then scatter)
// ---------------------------------------------------------------------------
void PardisoMPI::solve(const double* rhs, double* solution)
{
    if (!factorized_)
        throw std::runtime_error("solve() called before factorize()");

    // 1. Gather local RHS pieces onto rank 0.
    std::vector<double> global_rhs = gather_rhs(rhs);

    std::vector<double> global_sol;

    if (rank_ == 0) {
        global_sol.resize(n_, 0.0);

        int phase = 33;
        int nrhs  = 1;
        int error = 0;
        int idum  = 0;

        pardiso(pt_, &maxfct_, &mnum_, &mtype_, &phase,
                &n_, global_a_.data(), global_ia_.data(), global_ja_.data(),
                &idum, &nrhs, iparm_, &msglvl_,
                global_rhs.data(), global_sol.data(), &error);
        check_pardiso_error(error, "solve (phase 33)");
    }

    // 2. Scatter the global solution back to each rank.
    scatter_solution(rank_ == 0 ? global_sol.data() : nullptr, solution);
}

// ---------------------------------------------------------------------------
// solve_global  (phase 33 on rank 0, then broadcast)
// ---------------------------------------------------------------------------
void PardisoMPI::solve_global(const double* global_rhs, double* global_sol)
{
    if (!factorized_)
        throw std::runtime_error("solve_global() called before factorize()");

    if (rank_ == 0) {
        // PARDISO may modify the RHS buffer, so work on a copy.
        std::vector<double> rhs_copy(global_rhs, global_rhs + n_);

        int phase = 33;
        int nrhs  = 1;
        int error = 0;
        int idum  = 0;

        pardiso(pt_, &maxfct_, &mnum_, &mtype_, &phase,
                &n_, global_a_.data(), global_ia_.data(), global_ja_.data(),
                &idum, &nrhs, iparm_, &msglvl_,
                rhs_copy.data(), global_sol, &error);
        check_pardiso_error(error, "solve (phase 33)");
    }

    // Broadcast the full solution to all ranks.
    MPI_Bcast(global_sol, n_, MPI_DOUBLE, 0, comm_);
}

// ===================================================================
// Private helpers
// ===================================================================

// ---------------------------------------------------------------------------
// gather_matrix — rebuild full global CSR on rank 0
// ---------------------------------------------------------------------------
void PardisoMPI::gather_matrix()
{
    // Each rank sends:
    //   - number of local rows
    //   - local_rows_ (global indices)
    //   - local_ia_  (of size local_n + 1)
    //   - local_ja_
    //   - local_a_

    const int local_n  = static_cast<int>(local_rows_.size());
    const int local_nnz = static_cast<int>(local_ja_.size());

    // ------ Step 1: gather counts on rank 0 ------
    std::vector<int> all_local_n;
    std::vector<int> all_local_nnz;
    if (rank_ == 0) {
        all_local_n.resize(comm_size_);
        all_local_nnz.resize(comm_size_);
    }
    MPI_Gather(&local_n,   1, MPI_INT, all_local_n.data(),   1, MPI_INT, 0, comm_);
    MPI_Gather(&local_nnz, 1, MPI_INT, all_local_nnz.data(), 1, MPI_INT, 0, comm_);

    // ------ Step 2: gather local_rows_, local_ia_, local_ja_, local_a_ ------
    // Compute displacements for Gatherv.
    auto make_displs = [](const std::vector<int>& counts) {
        std::vector<int> d(counts.size());
        d[0] = 0;
        for (size_t i = 1; i < counts.size(); ++i)
            d[i] = d[i-1] + counts[i-1];
        return d;
    };

    // For rows and ia we need counts derived from all_local_n.
    std::vector<int> rows_counts, ia_counts;
    std::vector<int> rows_displs, ia_displs, nnz_displs;

    std::vector<int>    recv_rows, recv_ia, recv_ja;
    std::vector<double> recv_a;

    if (rank_ == 0) {
        rows_counts.resize(comm_size_);
        ia_counts.resize(comm_size_);
        for (int r = 0; r < comm_size_; ++r) {
            rows_counts[r] = all_local_n[r];
            ia_counts[r]   = all_local_n[r] + 1;
        }
        rows_displs = make_displs(rows_counts);
        ia_displs   = make_displs(ia_counts);
        nnz_displs  = make_displs(all_local_nnz);

        int total_rows = rows_displs.back() + rows_counts.back();
        int total_ia   = ia_displs.back()   + ia_counts.back();
        int total_nnz  = nnz_displs.back()  + all_local_nnz.back();

        recv_rows.resize(total_rows);
        recv_ia.resize(total_ia);
        recv_ja.resize(total_nnz);
        recv_a.resize(total_nnz);
    }

    MPI_Gatherv(local_rows_.data(), local_n, MPI_INT,
                recv_rows.data(),
                rank_ == 0 ? rows_counts.data() : nullptr,
                rank_ == 0 ? rows_displs.data() : nullptr,
                MPI_INT, 0, comm_);

    MPI_Gatherv(local_ia_.data(), local_n + 1, MPI_INT,
                recv_ia.data(),
                rank_ == 0 ? ia_counts.data() : nullptr,
                rank_ == 0 ? ia_displs.data() : nullptr,
                MPI_INT, 0, comm_);

    MPI_Gatherv(local_ja_.data(), local_nnz, MPI_INT,
                recv_ja.data(),
                rank_ == 0 ? all_local_nnz.data() : nullptr,
                rank_ == 0 ? nnz_displs.data() : nullptr,
                MPI_INT, 0, comm_);

    MPI_Gatherv(local_a_.data(), local_nnz, MPI_DOUBLE,
                recv_a.data(),
                rank_ == 0 ? all_local_nnz.data() : nullptr,
                rank_ == 0 ? nnz_displs.data() : nullptr,
                MPI_DOUBLE, 0, comm_);

    // ------ Step 3: rank 0 rebuilds the global CSR ------
    if (rank_ == 0) {
        // We received rows potentially out of global order from different
        // ranks. Build an index that maps global_row -> position in the
        // received flat arrays so we can reorder.

        // First, build per-received-row info.
        struct RowInfo {
            int global_row;
            int ja_offset;   // offset into recv_ja / recv_a
            int nnz;         // number of non-zeros in this row
        };

        std::vector<RowInfo> infos;
        infos.reserve(n_);

        for (int r = 0; r < comm_size_; ++r) {
            int rn  = all_local_n[r];
            int r_row_off = rows_displs[r];
            int r_ia_off  = ia_displs[r];
            // The local_ia for this rank is stored starting at recv_ia[r_ia_off].
            // Its values are 1-based local pointers; row k has entries from
            // recv_ia[r_ia_off + k] to recv_ia[r_ia_off + k + 1] (exclusive),
            // but they are local offsets. We need to map to the flat recv_ja.
            int r_nnz_off = nnz_displs[r];
            for (int li = 0; li < rn; ++li) {
                int local_start = recv_ia[r_ia_off + li]     - 1; // 0-based local
                int local_end   = recv_ia[r_ia_off + li + 1] - 1;
                RowInfo ri;
                ri.global_row = recv_rows[r_row_off + li];
                ri.ja_offset  = r_nnz_off + local_start;
                ri.nnz        = local_end - local_start;
                infos.push_back(ri);
            }
        }

        // Sort by global row index so the rebuilt CSR is in order.
        std::sort(infos.begin(), infos.end(),
                  [](const RowInfo& a, const RowInfo& b) {
                      return a.global_row < b.global_row;
                  });

        // Total NNZ.
        int total_nnz = 0;
        for (auto& ri : infos)
            total_nnz += ri.nnz;

        global_ia_.resize(n_ + 1);
        global_ja_.resize(total_nnz);
        global_a_.resize(total_nnz);

        int ptr = 1; // 1-based
        for (int i = 0; i < n_; ++i) {
            global_ia_[i] = ptr;
            const auto& ri = infos[i];
            assert(ri.global_row == i);
            std::memcpy(&global_ja_[ptr - 1], &recv_ja[ri.ja_offset],
                        ri.nnz * sizeof(int));
            std::memcpy(&global_a_[ptr - 1],  &recv_a[ri.ja_offset],
                        ri.nnz * sizeof(double));
            ptr += ri.nnz;
        }
        global_ia_[n_] = ptr;
    }
}

// ---------------------------------------------------------------------------
// gather_rhs — collect local RHS pieces onto rank 0
// ---------------------------------------------------------------------------
std::vector<double> PardisoMPI::gather_rhs(const double* local_rhs)
{
    const int local_n = static_cast<int>(local_rows_.size());

    // Gather counts.
    std::vector<int> counts;
    if (rank_ == 0) counts.resize(comm_size_);
    MPI_Gather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm_);

    std::vector<int> displs;
    std::vector<int> recv_rows_flat;
    std::vector<double> recv_vals;
    if (rank_ == 0) {
        displs.resize(comm_size_);
        displs[0] = 0;
        for (int r = 1; r < comm_size_; ++r)
            displs[r] = displs[r-1] + counts[r-1];
        int total = displs.back() + counts.back();
        recv_rows_flat.resize(total);
        recv_vals.resize(total);
    }

    // Gather global row indices (so we know where to place values).
    MPI_Gatherv(local_rows_.data(), local_n, MPI_INT,
                recv_rows_flat.data(),
                rank_ == 0 ? counts.data() : nullptr,
                rank_ == 0 ? displs.data() : nullptr,
                MPI_INT, 0, comm_);

    // Gather actual RHS values.
    MPI_Gatherv(local_rhs, local_n, MPI_DOUBLE,
                recv_vals.data(),
                rank_ == 0 ? counts.data() : nullptr,
                rank_ == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, comm_);

    std::vector<double> global_rhs;
    if (rank_ == 0) {
        global_rhs.resize(n_, 0.0);
        int total = static_cast<int>(recv_rows_flat.size());
        for (int k = 0; k < total; ++k) {
            global_rhs[recv_rows_flat[k]] = recv_vals[k];
        }
    }
    return global_rhs;
}

// ---------------------------------------------------------------------------
// scatter_solution — send solution entries back to owning ranks
// ---------------------------------------------------------------------------
void PardisoMPI::scatter_solution(const double* global_sol, double* local_sol)
{
    const int local_n = static_cast<int>(local_rows_.size());

    // Gather counts (same as in gather_rhs).
    std::vector<int> counts;
    if (rank_ == 0) counts.resize(comm_size_);
    MPI_Gather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm_);

    // On rank 0, reorder global_sol into the order that Scatterv expects:
    // for each rank r, the entries corresponding to that rank's local_rows_
    // in the same order they appear on that rank.
    //
    // We need to know each rank's local_rows_.  We already gathered them
    // during gather_matrix / gather_rhs; rather than caching, just gather
    // again (this is a small amount of data compared to the matrix).

    std::vector<int> displs;
    std::vector<int> recv_rows_flat;
    if (rank_ == 0) {
        displs.resize(comm_size_);
        displs[0] = 0;
        for (int r = 1; r < comm_size_; ++r)
            displs[r] = displs[r-1] + counts[r-1];
        int total = displs.back() + counts.back();
        recv_rows_flat.resize(total);
    }

    MPI_Gatherv(local_rows_.data(), local_n, MPI_INT,
                recv_rows_flat.data(),
                rank_ == 0 ? counts.data() : nullptr,
                rank_ == 0 ? displs.data() : nullptr,
                MPI_INT, 0, comm_);

    // Build the send buffer on rank 0 in Scatterv order.
    std::vector<double> send_buf;
    if (rank_ == 0) {
        int total = static_cast<int>(recv_rows_flat.size());
        send_buf.resize(total);
        for (int k = 0; k < total; ++k) {
            send_buf[k] = global_sol[recv_rows_flat[k]];
        }
    }

    MPI_Scatterv(rank_ == 0 ? send_buf.data() : nullptr,
                 rank_ == 0 ? counts.data() : nullptr,
                 rank_ == 0 ? displs.data() : nullptr,
                 MPI_DOUBLE,
                 local_sol, local_n, MPI_DOUBLE, 0, comm_);
}

// ---------------------------------------------------------------------------
// pardiso_cleanup — release PARDISO internal memory (phase -1)
// ---------------------------------------------------------------------------
void PardisoMPI::pardiso_cleanup()
{
    if (rank_ == 0 && (factorized_ || matrix_set_)) {
        int phase = -1;
        int nrhs  = 0;
        int error = 0;
        int idum  = 0;
        double ddum = 0.0;

        pardiso(pt_, &maxfct_, &mnum_, &mtype_, &phase,
                &n_,
                global_a_.empty() ? &ddum : global_a_.data(),
                global_ia_.empty() ? &idum : global_ia_.data(),
                global_ja_.empty() ? &idum : global_ja_.data(),
                &idum, &nrhs, iparm_, &msglvl_, &ddum, &ddum, &error);
        // Ignore errors during cleanup — we're in a destructor path.
    }

    // Clear gathered data.
    global_ia_.clear();
    global_ja_.clear();
    global_a_.clear();

    // Re-zero PARDISO handles.
    std::memset(pt_, 0, sizeof(pt_));

    factorized_ = false;
}
