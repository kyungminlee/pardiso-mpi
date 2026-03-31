#pragma once

#include <cstddef>
#include <filesystem>
#include <iosfwd>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

/// Compressed Sparse Row matrix with 1-based indexing (PARDISO convention).
struct CSRMatrix {
    int n   = 0;          ///< Number of rows.
    int m   = 0;          ///< Number of columns.
    int nnz = 0;          ///< Number of stored non-zeros.
    std::vector<int>    ia;   ///< Row pointers  (size n+1, 1-based).
    std::vector<int>    ja;   ///< Column indices (size nnz, 1-based).
    std::vector<double> a;    ///< Values         (size nnz).
};

// ---------------------------------------------------------------------------
// MappedFile — RAII read-only memory mapping
// ---------------------------------------------------------------------------

/// RAII wrapper around a read-only memory-mapped file (POSIX mmap).
/// Move-only.  The mapping stays valid until the object is destroyed.
class MappedFile {
public:
    /// Map the file at @p path into memory.
    /// @throws std::runtime_error if the file cannot be opened, is empty,
    ///         or mmap fails.
    explicit MappedFile(const std::filesystem::path& path);
    ~MappedFile();

    MappedFile(MappedFile&& other) noexcept;
    MappedFile& operator=(MappedFile&& other) noexcept;

    MappedFile(const MappedFile&)            = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    [[nodiscard]] const char*      data() const noexcept;
    [[nodiscard]] std::size_t      size() const noexcept;
    [[nodiscard]] std::string_view view() const noexcept;

private:
    void*       data_ = nullptr;
    std::size_t size_ = 0;
};

// ---------------------------------------------------------------------------
// MatrixMarketFile — mmap-backed lazy parser
// ---------------------------------------------------------------------------

/// Memory-mapped Matrix Market file with lazy data parsing.
///
/// The constructor mmaps the file and parses only the header (banner and
/// dimensions).  The numerical data is not touched until @c to_csr() is
/// called, at which point the COO entries are parsed directly from the
/// mapped memory (zero-copy, using @c std::from_chars) and converted to
/// 1-based CSR.  The result is cached so repeated calls are free.
///
/// @par Thread safety
/// Not thread-safe — concurrent calls to @c to_csr() / @c take_csr()
/// on the same instance are a data race.
class MatrixMarketFile {
public:
    enum class Field    { real, integer, pattern };
    enum class Symmetry { general, symmetric, skew_symmetric };

    /// Mmap the file and parse the header.
    /// @throws std::runtime_error with file path, line number, and
    ///         description on any header parse error.
    explicit MatrixMarketFile(const std::filesystem::path& path);

    // --- Header info (available immediately) ---

    [[nodiscard]] const std::filesystem::path& path()     const noexcept;
    [[nodiscard]] int      rows()     const noexcept;
    [[nodiscard]] int      cols()     const noexcept;
    [[nodiscard]] int      file_nnz() const noexcept;
    [[nodiscard]] Field    field()    const noexcept;
    [[nodiscard]] Symmetry symmetry() const noexcept;

    // --- Lazy data access ---

    /// Parse the COO data and return a 1-based CSR matrix.
    /// First call parses; subsequent calls return the cached result.
    [[nodiscard]] const CSRMatrix& to_csr() const;

    /// Move the CSR matrix out, invalidating the cache.
    /// The next call to @c to_csr() will re-parse.
    [[nodiscard]] CSRMatrix take_csr();

private:
    std::filesystem::path path_;
    MappedFile            mapping_;

    // Header (parsed eagerly in constructor).
    int      rows_      = 0;
    int      cols_      = 0;
    int      nnz_file_  = 0;
    Field    field_     = Field::real;
    Symmetry symmetry_  = Symmetry::general;

    // Position in the mapped data where COO entries begin.
    std::size_t data_offset_  = 0;
    std::size_t data_line_no_ = 0;

    mutable std::optional<CSRMatrix> cached_csr_;

    [[nodiscard]] CSRMatrix parse_data() const;
};

// ---------------------------------------------------------------------------
// Convenience free functions
// ---------------------------------------------------------------------------

/// Read a Matrix Market file into a 1-based CSR matrix (mmap + eager parse).
[[nodiscard]] CSRMatrix read_matrix_market(const std::filesystem::path& path);

/// Read from an already-open stream (no mmap — eager parse).
[[nodiscard]] CSRMatrix read_matrix_market(std::istream& in,
                                           std::string_view source_name = "<stream>");

/// Write a CSR matrix to a Matrix Market file (coordinate real general).
void write_matrix_market(const std::filesystem::path& path, const CSRMatrix& matrix);

/// Write to an already-open stream.
void write_matrix_market(std::ostream& out, const CSRMatrix& matrix);
