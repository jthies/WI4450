#pragma once

#include "operations.hpp"

void cg_solver(stencil3d const* op, int n, double* x, double const* b,
        double tol, int maxIter);
