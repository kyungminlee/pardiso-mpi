#include "pardiso_mpi.h"

#include <complex>

template class PardisoMPI<double>;
template class PardisoMPI<std::complex<double>>;
