#pragma once

#include "operations.hpp"
#include "cg_solver.hpp"

void time_integration_sequential_FE(stencil3d const* op, int n, double* x, double const* x_0,
        double  tol, double delta_t,   int  maxIter, int T,
        double* resNorm, int* numIter,
        int verbose=1);

void time_integration_sequential_BE(stencil3d const* op, int n, double* x, double const* x_0,
        double  tol, double delta_t,   int  maxIter, int T,
        double* resNorm, int* numIter,
        int verbose=1);

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

void time_integration_gmres(stencil3d const* L, int n, double* x0, const double* b,
        double epsilon, double delta_t, int maxIter, int T, 
        double* resNorm, int* numIter);
