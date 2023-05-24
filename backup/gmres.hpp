#pragma once

#include "operations.hpp"


void gmres(stencil3d const* L, const double* b, double* x0, int const maxIter, double epsilon, int n ,double delta_t,  int T, double* resNorm);