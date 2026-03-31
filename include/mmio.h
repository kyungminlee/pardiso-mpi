#pragma once

#include <cstdint>
#include <filesystem>
#include <iosfwd>
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
// Read
// ---------------------------------------------------------------------------

/// Read a Matrix Market file into a 1-based CSR matrix.
///
/// Supported banners:
///   - coordinate  real|integer|pattern  general|symmetric|skew-symmetric
///
/// For symmetric matrices both triangles are stored.  For skew-symmetric
/// matrices the transposed entry is negated.  Pattern matrices get value 1.
///
/// @throws std::runtime_error with file name, line number, and description.
[[nodiscard]] CSRMatrix read_matrix_market(const std::filesystem::path& path);

/// Read from an already-open stream (useful for embedded data / testing).
/// @param source_name  Label used in error messages (e.g. "<stdin>").
[[nodiscard]] CSRMatrix read_matrix_market(std::istream& in,
                                           std::string_view source_name = "<stream>");

// ---------------------------------------------------------------------------
// Write
// ---------------------------------------------------------------------------

/// Write a CSR matrix to a Matrix Market file (coordinate real general).
/// @throws std::runtime_error on I/O failure.
void write_matrix_market(const std::filesystem::path& path, const CSRMatrix& matrix);

/// Write to an already-open stream.
void write_matrix_market(std::ostream& out, const CSRMatrix& matrix);
