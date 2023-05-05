#pragma once

#include "operations.hpp"

// run Conjugate Gradient iterations to solve the linear system
// op*x=b, where op is the 7-point stencil representation of a linear
// operator. The function returns if the 2-norm of the residual reaches
// tol, or the number of iterations reaches maxIter. The residual norm
// is returned in *resNorm, the number of iterations in *numIter.
void time_integration_parallel_L_parallel_Jacobi(stencil3d const* op, int n, double* x, double const* x_0,
        double  tol, double delta_t,   int  maxIter, int T,
        double* resNorm, int* numIter,
        int verbose=1);

void time_integration_parallel_L_Jacobi(stencil3d const* op, int n, double* x, double const* x_0,
        double  tol, double delta_t,   int  maxIter, int T,
        double* resNorm, int* numIter,
        int verbose=1);

void time_integration_parallel_Jacobi(stencil3d const* op, int n, double* x, double const* x_0,
        double  tol, double delta_t,   int  maxIter, int T,
        double* resNorm, int* numIter,
        int verbose=1);

void time_integration_Jacobi(stencil3d const* op, int n, double* x, double const* x_0,
        double  tol, double delta_t,   int  maxIter, int T,
        double* resNorm, int* numIter,
        int verbose=1);

void time_integration_GMRES(stencil3d const* op, int n, double* x, double const* x_0,
        double  tol, double delta_t,   int  maxIter, int T,
        double* resNorm, int* numIter,
        int verbose=1);
