#pragma once
#include <string>
#include <vector>

struct CSRMatrix {
    int n;           // dimension (rows = cols for square)
    int m;           // columns
    int nnz;         // number of non-zeros
    std::vector<int> ia;      // row pointers (1-based, size n+1)
    std::vector<int> ja;      // column indices (1-based)
    std::vector<double> a;    // values
};

// Read a Matrix Market file and return CSR matrix (1-based indexing for PARDISO)
// Supports: coordinate real general, coordinate real symmetric
// For symmetric matrices, stores both upper and lower triangular parts
CSRMatrix read_matrix_market(const std::string& filename);

// Write a Matrix Market file from CSR matrix
void write_matrix_market(const std::string& filename, const CSRMatrix& matrix);
