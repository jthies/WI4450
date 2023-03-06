#include "operations.hpp"
#include <omp.h>

void init(int n, double* x, double value)
{
  // [...]
  for (int i = 0; i<n; i++)
  {
    x[i] = value;
  }
  return;
}

double dot(int n, double const* x, double const* y)
{
  // [...]
  double sum = 0.0;
  for (int i = 0; i<n; i++)
  {
    sum += x[i]*y[i];
  }
  return sum;
}

void axpby(int n, double a, double const* x, double b, double* y)
{
  // [...]
  for (int i = 0; i<n; i++)
  {
    y[i] = a*x[i] + b*y[i];
  }
  return;
}

//! apply a 7-point stencil to a vector
void apply_stencil3d(stencil3d const* S,
        double const* u, double* v)
{
  // [...]
  //v=S*u
  for (int i = 0; i < S->nx; i++)
  {
    for (int j = 0; j < S->ny; j++)
    {
      for (int k = 0; k < S->nz; k++)
      {
	  //grid i,j,k
	  v[S->index_c(i,j,k)] = S->value_c * u[S->index_c(i,j,k)];
	  //grid i-1,j,k
	  if (i > 0)
	  {
	    v[S->index_w(i,j,k)] += S->value_w * u[S->index_w(i,j,k)];
	  }
	  //grid i,j-1,k
	  if (j > 0)
	  {
	    v[S->index_s(i,j,k)] += S->value_s * u[S->index_s(i,j,k)];
	  }
	  //grid i,j,k-1
	  if (k > 0)
	  {
	    v[S->index_t(i,j,k)] += S->value_t * u[S->index_t(i,j,k)];
	  }
	  //grid i+1,j,k
	  if (i + 1 < S->nx)
	  {
	    v[S->index_e(i,j,k)] += S->value_e * u[S->index_e(i,j,k)];
	  }
	  //grid i,j+1,k
	  if (j + 1 < S->ny)
	  {
	    v[S->index_n(i,j,k)] += S->value_n * u[S->index_n(i,j,k)];
	  }
	  //grid i,j,k+1
	  if (k + 1 < S->nz)
	  {
	    v[S->index_b(i,j,k)] += S->value_b * u[S->index_b(i,j,k)];
	  }
      }
    }
  }
  return;
}

