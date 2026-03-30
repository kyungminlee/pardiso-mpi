#include "mmio.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <tuple>

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
namespace {

struct COOEntry {
    int row;  // 1-based
    int col;  // 1-based
    double val;
};

// Convert COO triplets to CSR with 1-based indexing.
// Assumes entries are not yet sorted.  Duplicate (i,j) pairs are summed.
CSRMatrix coo_to_csr(int n, int m, std::vector<COOEntry>& entries) {
    // Sort by (row, col) so that column indices within each row are ordered.
    std::sort(entries.begin(), entries.end(),
              [](const COOEntry& a, const COOEntry& b) {
                  return std::tie(a.row, a.col) < std::tie(b.row, b.col);
              });

    CSRMatrix csr;
    csr.n = n;
    csr.m = m;

    // Build row pointers (1-based indexing: ia[0] = 1)
    csr.ia.assign(static_cast<std::size_t>(n) + 1, 0);

    // First pass: count entries per row
    for (const auto& e : entries) {
        csr.ia[static_cast<std::size_t>(e.row)]++;
    }

    // Prefix sum to get row pointers (1-based)
    csr.ia[0] = 1;
    for (int i = 1; i <= n; ++i) {
        csr.ia[static_cast<std::size_t>(i)] += csr.ia[static_cast<std::size_t>(i) - 1];
    }

    int total_nnz = csr.ia[static_cast<std::size_t>(n)] - 1;
    csr.nnz = total_nnz;
    csr.ja.resize(static_cast<std::size_t>(total_nnz));
    csr.a.resize(static_cast<std::size_t>(total_nnz));

    // Second pass: place entries (using a working copy of ia)
    std::vector<int> work(csr.ia.begin(), csr.ia.end());
    for (const auto& e : entries) {
        int pos = work[static_cast<std::size_t>(e.row) - 1] - 1;  // convert to 0-based index
        csr.ja[static_cast<std::size_t>(pos)] = e.col;
        csr.a[static_cast<std::size_t>(pos)] = e.val;
        work[static_cast<std::size_t>(e.row) - 1]++;
    }

    return csr;
}

// Lowercase a string in-place and return it.
std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return s;
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// read_matrix_market
// ---------------------------------------------------------------------------
CSRMatrix read_matrix_market(const std::string& filename) {
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        throw std::runtime_error("read_matrix_market: cannot open file: " + filename);
    }

    // ---- Parse banner line ------------------------------------------------
    std::string banner_line;
    if (!std::getline(fin, banner_line)) {
        throw std::runtime_error("read_matrix_market: file is empty: " + filename);
    }

    std::istringstream banner_stream(banner_line);
    std::string token;
    std::vector<std::string> tokens;
    while (banner_stream >> token) {
        tokens.push_back(to_lower(token));
    }

    // Expected: %%matrixmarket matrix coordinate real general/symmetric
    if (tokens.size() < 5 || tokens[0] != "%%matrixmarket") {
        throw std::runtime_error(
            "read_matrix_market: invalid Matrix Market banner in: " + filename);
    }

    const std::string& object  = tokens[1];  // matrix
    const std::string& format  = tokens[2];  // coordinate
    const std::string& field   = tokens[3];  // real
    const std::string& symmetry = tokens[4]; // general | symmetric

    if (object != "matrix") {
        throw std::runtime_error(
            "read_matrix_market: unsupported object type '" + object + "' in: " + filename);
    }
    if (format != "coordinate") {
        throw std::runtime_error(
            "read_matrix_market: only coordinate format is supported, got '" + format +
            "' in: " + filename);
    }
    if (field != "real" && field != "integer" && field != "double") {
        throw std::runtime_error(
            "read_matrix_market: unsupported field type '" + field + "' in: " + filename);
    }
    if (symmetry != "general" && symmetry != "symmetric") {
        throw std::runtime_error(
            "read_matrix_market: unsupported symmetry type '" + symmetry + "' in: " + filename);
    }

    bool is_symmetric = (symmetry == "symmetric");

    // ---- Skip comment lines -----------------------------------------------
    std::string line;
    while (std::getline(fin, line)) {
        // Skip blank lines and comment lines
        if (line.empty() || line[0] == '%') {
            continue;
        }
        break;  // first non-comment, non-blank line is the size line
    }

    // ---- Read dimensions --------------------------------------------------
    int rows = 0, cols = 0, nnz_file = 0;
    {
        std::istringstream dim_stream(line);
        if (!(dim_stream >> rows >> cols >> nnz_file)) {
            throw std::runtime_error(
                "read_matrix_market: failed to read dimensions in: " + filename);
        }
    }

    if (rows <= 0 || cols <= 0 || nnz_file <= 0) {
        throw std::runtime_error(
            "read_matrix_market: invalid dimensions in: " + filename);
    }

    if (is_symmetric && rows != cols) {
        throw std::runtime_error(
            "read_matrix_market: symmetric matrix must be square in: " + filename);
    }

    // ---- Read COO entries -------------------------------------------------
    // Reserve generously for symmetric case (up to 2*nnz_file).
    std::vector<COOEntry> entries;
    entries.reserve(is_symmetric
                        ? static_cast<std::size_t>(2) * static_cast<std::size_t>(nnz_file)
                        : static_cast<std::size_t>(nnz_file));

    for (int k = 0; k < nnz_file; ++k) {
        int i = 0, j = 0;
        double v = 0.0;
        if (!(fin >> i >> j >> v)) {
            throw std::runtime_error(
                "read_matrix_market: premature end of data at entry " +
                std::to_string(k) + " in: " + filename);
        }

        entries.push_back({i, j, v});

        if (is_symmetric && i != j) {
            entries.push_back({j, i, v});
        }
    }

    // ---- Convert to CSR ---------------------------------------------------
    return coo_to_csr(rows, cols, entries);
}

// ---------------------------------------------------------------------------
// write_matrix_market
// ---------------------------------------------------------------------------
void write_matrix_market(const std::string& filename, const CSRMatrix& matrix) {
    std::ofstream fout(filename);
    if (!fout.is_open()) {
        throw std::runtime_error("write_matrix_market: cannot open file for writing: " + filename);
    }

    fout << "%%MatrixMarket matrix coordinate real general\n";
    fout << "% Generated by pardiso-mpi mmio writer\n";
    fout << matrix.n << " " << matrix.m << " " << matrix.nnz << "\n";

    fout.precision(17);  // full double precision
    for (int i = 0; i < matrix.n; ++i) {
        int row_start = matrix.ia[static_cast<std::size_t>(i)] - 1;      // convert to 0-based
        int row_end   = matrix.ia[static_cast<std::size_t>(i) + 1] - 1;
        for (int k = row_start; k < row_end; ++k) {
            fout << (i + 1) << " "
                 << matrix.ja[static_cast<std::size_t>(k)] << " "
                 << matrix.a[static_cast<std::size_t>(k)] << "\n";
        }
    }

    if (!fout) {
        throw std::runtime_error("write_matrix_market: write error for file: " + filename);
    }
}
