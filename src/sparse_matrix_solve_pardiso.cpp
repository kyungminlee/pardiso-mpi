#include "sparse_matrix_solve_pardiso.h"

#include <complex>

template struct Triplet<double>;
template struct Triplet<std::complex<double>>;

template class SparseMatrixSolvePardiso<double>;
template class SparseMatrixSolvePardiso<std::complex<double>>;
