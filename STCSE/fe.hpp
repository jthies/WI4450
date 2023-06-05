#include "operations.hpp"

void forward_euler(const int n,const int T,const int maxIter,const double epsilon,const double deltaT,const double* b,double* x,double* resNorm,const stencil3d* L);