#pragma once

#include "operations.hpp"

void gmres(stencil3d const* A, const double* b, double* x, int const maxIter, double epsilon, double* resNorm);