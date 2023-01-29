#include "operations.hpp"
#include <omp.h>

void init(int n, double* x, double value)
{
  // [...]
  return;
}

double dot(int n, double const* x, double const* y)
{
  // [...]
  return 0.0;
}

void axpby(int n, double a, double const* x, double b, double* y)
{
  // [...]
  return;
}

//! apply a 7-point stencil to a vector
void apply_stencil3d(stencil3d const* S,
        double const* u, double* v)
{
  // [...]
  return;
}

