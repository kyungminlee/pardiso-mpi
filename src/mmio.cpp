#include "mmio.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
namespace {

// ---- Error formatting -----------------------------------------------------

/// Build a human-readable error string:  "source:line: message"
[[nodiscard]] std::string format_error(std::string_view source,
                                       std::size_t line,
                                       std::string_view message) {
    std::string out;
    out.reserve(source.size() + 24 + message.size());
    out += source;
    out += ':';
    out += std::to_string(line);
    out += ": ";
    out += message;
    return out;
}

/// Convenience overload when there is no meaningful line number.
[[nodiscard]] std::string format_error(std::string_view source,
                                       std::string_view message) {
    std::string out;
    out.reserve(source.size() + 2 + message.size());
    out += source;
    out += ": ";
    out += message;
    return out;
}

// ---- String helpers -------------------------------------------------------

/// Lower-case an ASCII string in place and return it.
std::string& to_lower_inplace(std::string& s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

/// Return true if the line is empty or contains only whitespace / a comment.
[[nodiscard]] bool is_blank_or_comment(std::string_view line) {
    for (auto c : line) {
        if (c == '%') return true;
        if (!std::isspace(static_cast<unsigned char>(c))) return false;
    }
    return true;  // all whitespace
}

// ---- Banner enums ---------------------------------------------------------

enum class Field   { real, integer, pattern };
enum class Symmetry { general, symmetric, skew_symmetric };

// ---- COO → CSR conversion -------------------------------------------------

struct COOEntry {
    int    row;   // 1-based
    int    col;   // 1-based
    double val;
};

/// Convert COO triplets to 1-based CSR.
/// Duplicate (i,j) pairs are summed.  Column indices within each row are
/// sorted after construction.
[[nodiscard]] CSRMatrix coo_to_csr(int n, int m, std::vector<COOEntry>& entries) {
    // Sort by (row, col).
    std::sort(entries.begin(), entries.end(),
              [](const COOEntry& a, const COOEntry& b) {
                  return std::tie(a.row, a.col) < std::tie(b.row, b.col);
              });

    CSRMatrix csr;
    csr.n = n;
    csr.m = m;

    const auto N = static_cast<std::size_t>(n);

    // Count entries per row.
    csr.ia.assign(N + 1, 0);
    for (const auto& e : entries)
        csr.ia[static_cast<std::size_t>(e.row)]++;

    // Prefix sum → 1-based row pointers.
    csr.ia[0] = 1;
    for (std::size_t i = 1; i <= N; ++i)
        csr.ia[i] += csr.ia[i - 1];

    const int total_nnz = csr.ia[N] - 1;
    csr.nnz = total_nnz;
    csr.ja.resize(static_cast<std::size_t>(total_nnz));
    csr.a.resize(static_cast<std::size_t>(total_nnz));

    // Place entries using a working copy of ia.
    auto work = csr.ia;  // copy
    for (const auto& e : entries) {
        const auto ri  = static_cast<std::size_t>(e.row) - 1;
        const auto pos = static_cast<std::size_t>(work[ri] - 1);
        csr.ja[pos] = e.col;
        csr.a[pos]  = e.val;
        work[ri]++;
    }

    return csr;
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// read_matrix_market  (stream overload — core implementation)
// ---------------------------------------------------------------------------
CSRMatrix read_matrix_market(std::istream& in, std::string_view source_name) {
    std::string line;
    std::size_t line_no = 0;

    // Helper: read one non-blank, non-comment line. Returns false on EOF.
    auto next_data_line = [&]() -> bool {
        while (std::getline(in, line)) {
            ++line_no;
            if (!is_blank_or_comment(line)) return true;
        }
        return false;
    };

    // ---- Banner -----------------------------------------------------------
    if (!std::getline(in, line)) {
        throw std::runtime_error(format_error(source_name, "file is empty"));
    }
    ++line_no;

    std::istringstream banner_stream(line);
    std::string tok_id, tok_obj, tok_fmt, tok_field, tok_sym;
    if (!(banner_stream >> tok_id >> tok_obj >> tok_fmt >> tok_field >> tok_sym)) {
        throw std::runtime_error(format_error(
            source_name, line_no,
            "invalid Matrix Market banner (expected 5 tokens)"));
    }

    to_lower_inplace(tok_id);
    to_lower_inplace(tok_obj);
    to_lower_inplace(tok_fmt);
    to_lower_inplace(tok_field);
    to_lower_inplace(tok_sym);

    if (tok_id != "%%matrixmarket") {
        throw std::runtime_error(format_error(
            source_name, line_no,
            "banner must start with %%MatrixMarket"));
    }
    if (tok_obj != "matrix") {
        throw std::runtime_error(format_error(
            source_name, line_no,
            "unsupported object '" + tok_obj + "' (only 'matrix' is supported)"));
    }
    if (tok_fmt != "coordinate") {
        throw std::runtime_error(format_error(
            source_name, line_no,
            "unsupported format '" + tok_fmt + "' (only 'coordinate' is supported)"));
    }

    // Field type.
    Field field{};
    if      (tok_field == "real" || tok_field == "double") field = Field::real;
    else if (tok_field == "integer")                       field = Field::integer;
    else if (tok_field == "pattern")                       field = Field::pattern;
    else if (tok_field == "complex") {
        throw std::runtime_error(format_error(
            source_name, line_no,
            "complex field type is not supported (CSRMatrix stores real values)"));
    } else {
        throw std::runtime_error(format_error(
            source_name, line_no,
            "unknown field type '" + tok_field + "'"));
    }

    // Symmetry type.
    Symmetry symmetry{};
    if      (tok_sym == "general")        symmetry = Symmetry::general;
    else if (tok_sym == "symmetric")      symmetry = Symmetry::symmetric;
    else if (tok_sym == "skew-symmetric") symmetry = Symmetry::skew_symmetric;
    else if (tok_sym == "hermitian") {
        throw std::runtime_error(format_error(
            source_name, line_no,
            "hermitian symmetry is not supported (CSRMatrix stores real values)"));
    } else {
        throw std::runtime_error(format_error(
            source_name, line_no,
            "unknown symmetry type '" + tok_sym + "'"));
    }

    const bool expand_symmetric = (symmetry != Symmetry::general);

    // ---- Dimensions -------------------------------------------------------
    if (!next_data_line()) {
        throw std::runtime_error(format_error(
            source_name, "unexpected end of file before dimensions line"));
    }

    int rows = 0, cols = 0, nnz_file = 0;
    {
        std::istringstream dim_stream(line);
        if (!(dim_stream >> rows >> cols >> nnz_file)) {
            throw std::runtime_error(format_error(
                source_name, line_no,
                "failed to parse dimensions (expected: rows cols nnz)"));
        }
    }

    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error(format_error(
            source_name, line_no,
            "invalid dimensions: rows=" + std::to_string(rows) +
            " cols=" + std::to_string(cols)));
    }
    if (nnz_file < 0) {
        throw std::runtime_error(format_error(
            source_name, line_no,
            "negative nnz count: " + std::to_string(nnz_file)));
    }
    if (expand_symmetric && rows != cols) {
        throw std::runtime_error(format_error(
            source_name, line_no,
            "symmetric / skew-symmetric matrix must be square, but got " +
            std::to_string(rows) + " x " + std::to_string(cols)));
    }

    // ---- Read COO entries -------------------------------------------------
    std::vector<COOEntry> entries;
    entries.reserve(expand_symmetric
                        ? static_cast<std::size_t>(nnz_file) * 2
                        : static_cast<std::size_t>(nnz_file));

    for (int k = 0; k < nnz_file; ++k) {
        if (!next_data_line()) {
            throw std::runtime_error(format_error(
                source_name, line_no,
                "premature end of data: expected " + std::to_string(nnz_file) +
                " entries but got " + std::to_string(k)));
        }

        std::istringstream entry_stream(line);
        int i = 0, j = 0;
        double v = 1.0;  // default for pattern matrices

        if (!(entry_stream >> i >> j)) {
            throw std::runtime_error(format_error(
                source_name, line_no,
                "failed to parse row/col indices at entry " + std::to_string(k + 1)));
        }

        if (field != Field::pattern) {
            if (!(entry_stream >> v)) {
                throw std::runtime_error(format_error(
                    source_name, line_no,
                    "failed to parse value at entry " + std::to_string(k + 1)));
            }
        }

        // Bounds check.
        if (i < 1 || i > rows || j < 1 || j > cols) {
            throw std::runtime_error(format_error(
                source_name, line_no,
                "index out of range: (" + std::to_string(i) + ", " +
                std::to_string(j) + ") not in [1.." + std::to_string(rows) +
                "] x [1.." + std::to_string(cols) + "]"));
        }

        entries.push_back({i, j, v});

        if (expand_symmetric && i != j) {
            const double v_sym = (symmetry == Symmetry::skew_symmetric) ? -v : v;
            entries.push_back({j, i, v_sym});
        }
    }

    // ---- Convert to CSR ---------------------------------------------------
    return coo_to_csr(rows, cols, entries);
}

// ---------------------------------------------------------------------------
// read_matrix_market  (path overload)
// ---------------------------------------------------------------------------
CSRMatrix read_matrix_market(const std::filesystem::path& path) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        throw std::runtime_error(
            format_error(path.string(), "cannot open file for reading"));
    }
    return read_matrix_market(fin, path.string());
}

// ---------------------------------------------------------------------------
// write_matrix_market  (stream overload — core implementation)
// ---------------------------------------------------------------------------
void write_matrix_market(std::ostream& out, const CSRMatrix& matrix) {
    out << "%%MatrixMarket matrix coordinate real general\n"
        << "% Generated by pardiso-mpi mmio writer\n"
        << matrix.n << ' ' << matrix.m << ' ' << matrix.nnz << '\n';

    out.precision(17);
    for (int i = 0; i < matrix.n; ++i) {
        const auto row_begin = static_cast<std::size_t>(matrix.ia[static_cast<std::size_t>(i)] - 1);
        const auto row_end   = static_cast<std::size_t>(matrix.ia[static_cast<std::size_t>(i) + 1] - 1);
        for (auto k = row_begin; k < row_end; ++k) {
            out << (i + 1) << ' ' << matrix.ja[k] << ' ' << matrix.a[k] << '\n';
        }
    }

    if (!out) {
        throw std::runtime_error("write_matrix_market: stream write failed");
    }
}

// ---------------------------------------------------------------------------
// write_matrix_market  (path overload)
// ---------------------------------------------------------------------------
void write_matrix_market(const std::filesystem::path& path, const CSRMatrix& matrix) {
    std::ofstream fout(path);
    if (!fout.is_open()) {
        throw std::runtime_error(
            format_error(path.string(), "cannot open file for writing"));
    }
    write_matrix_market(fout, matrix);
    if (!fout) {
        throw std::runtime_error(
            format_error(path.string(), "write error"));
    }
}
