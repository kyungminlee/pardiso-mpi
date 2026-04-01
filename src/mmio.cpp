#include "mmio.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// ===========================================================================
// Internal helpers
// ===========================================================================
namespace {

// ---- Error formatting -----------------------------------------------------

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

[[nodiscard]] std::string format_error(std::string_view source,
                                       std::string_view message) {
    std::string out;
    out.reserve(source.size() + 2 + message.size());
    out += source;
    out += ": ";
    out += message;
    return out;
}

// ---- Lightweight scanners on raw char ranges ------------------------------

/// Advance past spaces and tabs.
const char* skip_blanks(const char* p, const char* end) noexcept {
    while (p < end && (*p == ' ' || *p == '\t')) ++p;
    return p;
}

/// Advance past the current line (past the next '\n', or to end).
const char* skip_line(const char* p, const char* end) noexcept {
    while (p < end && *p != '\n') ++p;
    if (p < end) ++p;          // skip the '\n' itself
    return p;
}

/// Read a line from [p, end) as a string_view (stripping trailing \r).
/// Advances p past the '\n'.  Returns empty view at EOF.
std::string_view read_line(const char*& p, const char* end) noexcept {
    if (p >= end) return {};
    const char* start = p;
    while (p < end && *p != '\n') ++p;
    const char* line_end = p;
    if (p < end) ++p;          // skip '\n'
    // strip trailing '\r'
    if (line_end > start && *(line_end - 1) == '\r') --line_end;
    return {start, static_cast<std::size_t>(line_end - start)};
}

/// Return true if the line is blank or a comment (starts with %).
[[nodiscard]] bool is_blank_or_comment(std::string_view line) noexcept {
    for (auto c : line) {
        if (c == '%') return true;
        if (!std::isspace(static_cast<unsigned char>(c))) return false;
    }
    return true;
}

/// Lower-case an ASCII string in place.
std::string& to_lower_inplace(std::string& s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

// ---- Zero-copy numeric parsers (from_chars on mmap'd data) ----------------

int parse_int(const char*& p, const char* end,
              std::string_view source, std::size_t line_no) {
    p = skip_blanks(p, end);
    int val = 0;
    auto [ptr, ec] = std::from_chars(p, end, val);
    if (ec != std::errc{}) {
        throw std::runtime_error(format_error(source, line_no,
                                              "failed to parse integer"));
    }
    p = ptr;
    return val;
}

double parse_double(const char*& p, const char* end,
                    std::string_view source, std::size_t line_no) {
    p = skip_blanks(p, end);
    double val = 0.0;
    auto [ptr, ec] = std::from_chars(p, end, val);
    if (ec != std::errc{}) {
        throw std::runtime_error(format_error(source, line_no,
                                              "failed to parse floating-point value"));
    }
    p = ptr;
    return val;
}

// ---- COO → CSR conversion (shared by both paths) -------------------------

struct COOEntry {
    int    row;   // 1-based
    int    col;   // 1-based
    double val;
};

[[nodiscard]] CSRMatrix coo_to_csr(int n, int m,
                                    std::vector<COOEntry>& entries) {
    std::sort(entries.begin(), entries.end(),
              [](const COOEntry& a, const COOEntry& b) {
                  return std::tie(a.row, a.col) < std::tie(b.row, b.col);
              });

    CSRMatrix csr;
    csr.n = n;
    csr.m = m;

    const auto N = static_cast<std::size_t>(n);

    csr.ia.assign(N + 1, 0);
    for (const auto& e : entries)
        csr.ia[static_cast<std::size_t>(e.row)]++;

    csr.ia[0] = 1;
    for (std::size_t i = 1; i <= N; ++i)
        csr.ia[i] += csr.ia[i - 1];

    const int total_nnz = csr.ia[N] - 1;
    csr.nnz = total_nnz;
    csr.ja.resize(static_cast<std::size_t>(total_nnz));
    csr.a.resize(static_cast<std::size_t>(total_nnz));

    auto work = csr.ia;
    for (const auto& e : entries) {
        const auto ri  = static_cast<std::size_t>(e.row) - 1;
        const auto pos = static_cast<std::size_t>(work[ri] - 1);
        csr.ja[pos] = e.col;
        csr.a[pos]  = e.val;
        work[ri]++;
    }

    return csr;
}

// ---- Shared banner / dimension parser for the stream path -----------------

using Field    = MatrixMarketFile::Field;
using Symmetry = MatrixMarketFile::Symmetry;

struct HeaderInfo {
    int      rows;
    int      cols;
    int      nnz_file;
    Field    field;
    Symmetry symmetry;
};

/// Parse banner tokens (already lower-cased) and return field + symmetry.
void parse_banner_tokens(const std::string& tok_obj,
                         const std::string& tok_fmt,
                         const std::string& tok_field,
                         const std::string& tok_sym,
                         std::string_view source,
                         std::size_t line_no,
                         Field& field_out,
                         Symmetry& sym_out) {
    if (tok_obj != "matrix") {
        throw std::runtime_error(format_error(
            source, line_no,
            "unsupported object '" + tok_obj + "' (only 'matrix' is supported)"));
    }
    if (tok_fmt != "coordinate") {
        throw std::runtime_error(format_error(
            source, line_no,
            "unsupported format '" + tok_fmt + "' (only 'coordinate' is supported)"));
    }

    if      (tok_field == "real" || tok_field == "double") field_out = Field::real;
    else if (tok_field == "integer")                       field_out = Field::integer;
    else if (tok_field == "pattern")                       field_out = Field::pattern;
    else if (tok_field == "complex") {
        throw std::runtime_error(format_error(
            source, line_no,
            "complex field type is not supported (CSRMatrix stores real values)"));
    } else {
        throw std::runtime_error(format_error(
            source, line_no, "unknown field type '" + tok_field + "'"));
    }

    if      (tok_sym == "general")        sym_out = Symmetry::general;
    else if (tok_sym == "symmetric")      sym_out = Symmetry::symmetric;
    else if (tok_sym == "skew-symmetric") sym_out = Symmetry::skew_symmetric;
    else if (tok_sym == "hermitian") {
        throw std::runtime_error(format_error(
            source, line_no,
            "hermitian symmetry is not supported (CSRMatrix stores real values)"));
    } else {
        throw std::runtime_error(format_error(
            source, line_no, "unknown symmetry type '" + tok_sym + "'"));
    }
}

}  // anonymous namespace

// ===========================================================================
// MappedFile
// ===========================================================================

MappedFile::MappedFile(const std::filesystem::path& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd == -1) {
        throw std::runtime_error(
            "MappedFile: cannot open '" + path.string() +
            "': " + std::strerror(errno));
    }

    struct stat st{};
    if (::fstat(fd, &st) == -1) {
        auto saved = errno;
        ::close(fd);
        throw std::runtime_error(
            "MappedFile: fstat failed for '" + path.string() +
            "': " + std::strerror(saved));
    }

    size_ = static_cast<std::size_t>(st.st_size);
    if (size_ == 0) {
        ::close(fd);
        throw std::runtime_error(
            "MappedFile: file is empty: '" + path.string() + "'");
    }

    data_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);  // fd is no longer needed after mmap

    if (data_ == MAP_FAILED) {
        auto saved = errno;
        data_ = nullptr;
        throw std::runtime_error(
            "MappedFile: mmap failed for '" + path.string() +
            "': " + std::strerror(saved));
    }

    ::madvise(data_, size_, MADV_SEQUENTIAL);
}

MappedFile::~MappedFile() {
    if (data_) ::munmap(data_, size_);
}

MappedFile::MappedFile(MappedFile&& other) noexcept
    : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
    if (this != &other) {
        if (data_) ::munmap(data_, size_);
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

const char*      MappedFile::data() const noexcept { return static_cast<const char*>(data_); }
std::size_t      MappedFile::size() const noexcept { return size_; }
std::string_view MappedFile::view() const noexcept { return {data(), size_}; }

// ===========================================================================
// MatrixMarketFile
// ===========================================================================

MatrixMarketFile::MatrixMarketFile(const std::filesystem::path& path)
    : path_(path), mapping_(path) {

    const char* p   = mapping_.data();
    const char* end = p + mapping_.size();
    std::size_t line_no = 0;

    // ---- Banner line ------------------------------------------------------
    auto banner = read_line(p, end);
    ++line_no;
    if (banner.empty()) {
        throw std::runtime_error(format_error(path_.string(), "file is empty"));
    }

    // Tokenize the banner.  We need mutable strings for to_lower_inplace,
    // but the banner is a single short line so the copy is negligible.
    std::istringstream bss{std::string(banner)};
    std::string tok_id, tok_obj, tok_fmt, tok_field, tok_sym;
    if (!(bss >> tok_id >> tok_obj >> tok_fmt >> tok_field >> tok_sym)) {
        throw std::runtime_error(format_error(
            path_.string(), line_no,
            "invalid Matrix Market banner (expected 5 tokens)"));
    }

    to_lower_inplace(tok_id);
    to_lower_inplace(tok_obj);
    to_lower_inplace(tok_fmt);
    to_lower_inplace(tok_field);
    to_lower_inplace(tok_sym);

    if (tok_id != "%%matrixmarket") {
        throw std::runtime_error(format_error(
            path_.string(), line_no,
            "banner must start with %%MatrixMarket"));
    }

    parse_banner_tokens(tok_obj, tok_fmt, tok_field, tok_sym,
                        path_.string(), line_no, field_, symmetry_);

    // ---- Skip comments / blanks to the dimensions line --------------------
    std::string_view dim_line;
    while (p < end) {
        dim_line = read_line(p, end);
        ++line_no;
        if (!is_blank_or_comment(dim_line)) break;
        dim_line = {};
    }
    if (dim_line.empty()) {
        throw std::runtime_error(format_error(
            path_.string(), "unexpected end of file before dimensions line"));
    }

    // Parse dimensions from the string_view using from_chars.
    {
        const char* dp   = dim_line.data();
        const char* dend = dp + dim_line.size();
        rows_     = parse_int(dp, dend, path_.string(), line_no);
        cols_     = parse_int(dp, dend, path_.string(), line_no);
        nnz_file_ = parse_int(dp, dend, path_.string(), line_no);
    }

    if (rows_ <= 0 || cols_ <= 0) {
        throw std::runtime_error(format_error(
            path_.string(), line_no,
            "invalid dimensions: rows=" + std::to_string(rows_) +
            " cols=" + std::to_string(cols_)));
    }
    if (nnz_file_ < 0) {
        throw std::runtime_error(format_error(
            path_.string(), line_no,
            "negative nnz count: " + std::to_string(nnz_file_)));
    }
    if (symmetry_ != Symmetry::general && rows_ != cols_) {
        throw std::runtime_error(format_error(
            path_.string(), line_no,
            "symmetric / skew-symmetric matrix must be square, but got " +
            std::to_string(rows_) + " x " + std::to_string(cols_)));
    }

    // Record where the data section starts.
    data_offset_  = static_cast<std::size_t>(p - mapping_.data());
    data_line_no_ = line_no;
}

// ---- Accessors ------------------------------------------------------------

const std::filesystem::path& MatrixMarketFile::path()     const noexcept { return path_; }
int      MatrixMarketFile::rows()     const noexcept { return rows_; }
int      MatrixMarketFile::cols()     const noexcept { return cols_; }
int      MatrixMarketFile::file_nnz() const noexcept { return nnz_file_; }
auto     MatrixMarketFile::field()    const noexcept -> Field    { return field_; }
auto     MatrixMarketFile::symmetry() const noexcept -> Symmetry { return symmetry_; }

// ---- Lazy parse -----------------------------------------------------------

const CSRMatrix& MatrixMarketFile::to_csr() const {
    if (!cached_csr_)
        cached_csr_ = parse_data();
    return *cached_csr_;
}

CSRMatrix MatrixMarketFile::take_csr() {
    if (!cached_csr_)
        cached_csr_ = parse_data();
    CSRMatrix out = std::move(*cached_csr_);
    cached_csr_.reset();
    return out;
}

CSRMatrix MatrixMarketFile::parse_data() const {
    const bool expand = (symmetry_ != Symmetry::general);
    const bool negate = (symmetry_ == Symmetry::skew_symmetric);
    const bool is_pattern = (field_ == Field::pattern);

    std::vector<COOEntry> entries;
    entries.reserve(expand ? static_cast<std::size_t>(nnz_file_) * 2
                           : static_cast<std::size_t>(nnz_file_));

    const char* p   = mapping_.data() + data_offset_;
    const char* end = mapping_.data() + mapping_.size();
    std::size_t line_no = data_line_no_;
    const auto src = path_.string();

    int entries_read = 0;
    while (entries_read < nnz_file_) {
        // Skip blank / comment lines.
        while (p < end) {
            const char* lp = skip_blanks(p, end);
            if (lp < end && *lp != '\n' && *lp != '\r' && *lp != '%')
                break;
            p = skip_line(p, end);
            ++line_no;
        }

        if (p >= end) {
            throw std::runtime_error(format_error(
                src, line_no,
                "premature end of data: expected " + std::to_string(nnz_file_) +
                " entries but got " + std::to_string(entries_read)));
        }

        ++line_no;

        int i = parse_int(p, end, src, line_no);
        int j = parse_int(p, end, src, line_no);
        double v = 1.0;
        if (!is_pattern)
            v = parse_double(p, end, src, line_no);

        // Advance past the rest of this line.
        p = skip_line(p, end);

        if (i < 1 || i > rows_ || j < 1 || j > cols_) {
            throw std::runtime_error(format_error(
                src, line_no,
                "index out of range: (" + std::to_string(i) + ", " +
                std::to_string(j) + ") not in [1.." + std::to_string(rows_) +
                "] x [1.." + std::to_string(cols_) + "]"));
        }

        entries.push_back({i, j, v});
        if (expand && i != j)
            entries.push_back({j, i, negate ? -v : v});

        ++entries_read;
    }

    return coo_to_csr(rows_, cols_, entries);
}

// ===========================================================================
// Free functions — file path overloads
// ===========================================================================

CSRMatrix read_matrix_market(const std::filesystem::path& path) {
    MatrixMarketFile mmf(path);
    return mmf.take_csr();
}

// ===========================================================================
// Free functions — stream overloads (no mmap, eager parse)
// ===========================================================================

CSRMatrix read_matrix_market(std::istream& in, std::string_view source_name) {
    std::string line;
    std::size_t line_no = 0;

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

    std::istringstream bss(line);
    std::string tok_id, tok_obj, tok_fmt, tok_field, tok_sym;
    if (!(bss >> tok_id >> tok_obj >> tok_fmt >> tok_field >> tok_sym)) {
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

    Field field{};
    Symmetry symmetry{};
    parse_banner_tokens(tok_obj, tok_fmt, tok_field, tok_sym,
                        source_name, line_no, field, symmetry);

    const bool expand     = (symmetry != Symmetry::general);
    const bool negate     = (symmetry == Symmetry::skew_symmetric);
    const bool is_pattern = (field == Field::pattern);

    // ---- Dimensions -------------------------------------------------------
    if (!next_data_line()) {
        throw std::runtime_error(format_error(
            source_name, "unexpected end of file before dimensions line"));
    }

    int rows = 0, cols = 0, nnz_file = 0;
    {
        std::istringstream ds(line);
        if (!(ds >> rows >> cols >> nnz_file)) {
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
    if (expand && rows != cols) {
        throw std::runtime_error(format_error(
            source_name, line_no,
            "symmetric / skew-symmetric matrix must be square, but got " +
            std::to_string(rows) + " x " + std::to_string(cols)));
    }

    // ---- COO entries ------------------------------------------------------
    std::vector<COOEntry> entries;
    entries.reserve(expand ? static_cast<std::size_t>(nnz_file) * 2
                           : static_cast<std::size_t>(nnz_file));

    for (int k = 0; k < nnz_file; ++k) {
        if (!next_data_line()) {
            throw std::runtime_error(format_error(
                source_name, line_no,
                "premature end of data: expected " + std::to_string(nnz_file) +
                " entries but got " + std::to_string(k)));
        }

        std::istringstream es(line);
        int i = 0, j = 0;
        double v = 1.0;

        if (!(es >> i >> j)) {
            throw std::runtime_error(format_error(
                source_name, line_no,
                "failed to parse row/col at entry " + std::to_string(k + 1)));
        }
        if (!is_pattern && !(es >> v)) {
            throw std::runtime_error(format_error(
                source_name, line_no,
                "failed to parse value at entry " + std::to_string(k + 1)));
        }

        if (i < 1 || i > rows || j < 1 || j > cols) {
            throw std::runtime_error(format_error(
                source_name, line_no,
                "index out of range: (" + std::to_string(i) + ", " +
                std::to_string(j) + ") not in [1.." + std::to_string(rows) +
                "] x [1.." + std::to_string(cols) + "]"));
        }

        entries.push_back({i, j, v});
        if (expand && i != j)
            entries.push_back({j, i, negate ? -v : v});
    }

    return coo_to_csr(rows, cols, entries);
}

// ===========================================================================
// Write
// ===========================================================================

void write_matrix_market(std::ostream& out, const CSRMatrix& matrix) {
    out << "%%MatrixMarket matrix coordinate real general\n"
        << "% Generated by pardiso-mpi mmio writer\n"
        << matrix.n << ' ' << matrix.m << ' ' << matrix.nnz << '\n';

    out.precision(17);
    for (int i = 0; i < matrix.n; ++i) {
        const auto begin = static_cast<std::size_t>(matrix.ia[static_cast<std::size_t>(i)] - 1);
        const auto end   = static_cast<std::size_t>(matrix.ia[static_cast<std::size_t>(i) + 1] - 1);
        for (auto k = begin; k < end; ++k) {
            out << (i + 1) << ' ' << matrix.ja[k] << ' ' << matrix.a[k] << '\n';
        }
    }

    if (!out) {
        throw std::runtime_error("write_matrix_market: stream write failed");
    }
}

void write_matrix_market(const std::filesystem::path& path, const CSRMatrix& matrix) {
    std::ofstream fout(path);
    if (!fout.is_open()) {
        throw std::runtime_error(
            format_error(path.string(), "cannot open file for writing"));
    }
    write_matrix_market(fout, matrix);
    if (!fout) {
        throw std::runtime_error(format_error(path.string(), "write error"));
    }
}
