#include "operations.hpp"

void perturb_gmres(const int n,const int T,const int maxIter,const double epsilon, const double deltaT,const double* b, double* x0, double* resNorm, const stencil3d* L);

void jacobi_gmres(const int n,const int T,const int maxIter,const double epsilon, const double deltaT,const double* b, double* x0, double* resNorm, const stencil3d* L);