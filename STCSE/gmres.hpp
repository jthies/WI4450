#include "operations.hpp"

void givens(const int j, const int maxIter, double* cos, double* sin, double* H);

void backward_substitution(const int iter, const int maxIter, const double* e_1,const double* H, double* y);

void gmres(const int n,const int T,const int maxIter,const double epsilon, const double deltaT,const double* b, const double* x0, double* resNorm, const stencil3d* L);