#include "operations.hpp"
#include <omp.h>

void init(int n, double* x, double value)
{
  #pragma omp parallel for schedule(static)
  for (int i = 0; i<n; i++)
  {
    x[i] = value;
  }
  return;
}

double dot(int n, double const* x, double const* y)
{
  double sum = 0.0;
  #pragma omp parallel for reduction(+:sum) schedule(static)
  for (int i = 0; i<n; i++)
  {
    sum += x[i]*y[i];
  }
  return sum;
}

void axpby(int n, double a, double const* x, double b, double* y)
{
  #pragma omp parallel for schedule(static)
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
  //v=S*u: v,u vectors and S a 7-point stencil
  
  //interior points, k=1,...,nz-2, j=1,...,ny-2, i=1,...,nx-2
  #pragma omp parallel for schedule(static) //collapse(3) //<-for task 6 hw1, uncomment
  for (int k = 1; k < S->nz - 1; k++)
  {
    for (int j = 1; j < S->ny - 1; j++)
    {
      for (int i = 1; i < S->nx - 1; i++)
      {
        v[S->index_c(i, j, k)]  = S->value_b * u[S->index_b(i, j, k)]
                                + S->value_s * u[S->index_s(i, j, k)]
                                + S->value_w * u[S->index_w(i, j, k)]
                                + S->value_c * u[S->index_c(i, j, k)]
                                + S->value_e * u[S->index_e(i, j, k)]
                                + S->value_n * u[S->index_n(i, j, k)]
                                + S->value_t * u[S->index_t(i, j, k)];
      }
    }
  }
  
  //boundary points
  //face: k = 0, j=1,...,ny-2, i=1,...,nx-2
  #pragma omp parallel for schedule(static) //collapse(2) //<-for task 6 hw1, uncomment
  for (int j = 1; j < S->ny - 1; j++)
  {
    for (int i = 1; i < S->nx - 1; i++)
    {
        v[S->index_c(i, j, 0)]   = S->value_s * u[S->index_s(i, j, 0)]
                                + S->value_w * u[S->index_w(i, j, 0)]
                                + S->value_c * u[S->index_c(i, j, 0)]
                                + S->value_e * u[S->index_e(i, j, 0)]
                                + S->value_n * u[S->index_n(i, j, 0)]
                                + S->value_t * u[S->index_t(i, j, 0)];
    }
  }

  //face: k = nz-1, j=1,...,ny-2, i=1,...,nx-2
  #pragma omp parallel for schedule(static) //collapse(2) //<-for task 6 hw1, uncomment
  for (int j = 1; j < S->ny - 1; j++)
  {
    for (int i = 1; i < S->nx - 1; i++)
    {
        v[S->index_c(i, j, S->nz - 1)]  = S->value_b * u[S->index_b(i, j, S->nz - 1)]
                                        + S->value_s * u[S->index_s(i, j, S->nz - 1)]
                                        + S->value_w * u[S->index_w(i, j, S->nz - 1)]
                                        + S->value_c * u[S->index_c(i, j, S->nz - 1)]
                                        + S->value_e * u[S->index_e(i, j, S->nz - 1)]
                                        + S->value_n * u[S->index_n(i, j, S->nz - 1)];
    }
  }
  
  //face: k=1,...,nz-2, j=0, i=1,...,nx-2
  #pragma omp parallel for schedule(static) //collapse(2) //<-for task 6 hw1, uncomment
  for (int k = 1; k < S->nz - 1; k++)
  {
    for (int i = 1; i < S->nx - 1; i++)
    {
        v[S->index_c(i, 0, k)]  = S->value_b * u[S->index_b(i, 0, k)]
                                + S->value_w * u[S->index_w(i, 0, k)]
                                + S->value_c * u[S->index_c(i, 0, k)]
                                + S->value_e * u[S->index_e(i, 0, k)]
                                + S->value_n * u[S->index_n(i, 0, k)]
                                + S->value_t * u[S->index_t(i, 0, k)];
    }
  }
        

  //face: k=1,...,nz-2, j=ny-1, i=1,...,nx-2
  #pragma omp parallel for schedule(static) //collapse(2) //<-for task 6 hw1, uncomment
  for (int k = 1; k < S->nz - 1; k++)
  {
    for (int i = 1; i < S->nx - 1; i++)
    {
        v[S->index_c(i, S->ny - 1, k)]  = S->value_b * u[S->index_b(i, S->ny - 1, k)]
                                        + S->value_s * u[S->index_s(i, S->ny - 1, k)]
                                        + S->value_w * u[S->index_w(i, S->ny - 1, k)]
                                        + S->value_c * u[S->index_c(i, S->ny - 1, k)]
                                        + S->value_e * u[S->index_e(i, S->ny - 1, k)]
                                        + S->value_t * u[S->index_t(i, S->ny - 1, k)];
    }
  }

  //face: k=1,...,nz-2, j=1,...,ny-2, i=0
  #pragma omp parallel for schedule(static) //collapse(2) //<-for task 6 hw1, uncomment
  for (int k = 1; k < S->nz - 1; k++)
  {
    for (int j = 1; j < S->ny - 1; j++)
    {
        v[S->index_c(0, j, k)]  = S->value_b * u[S->index_b(0, j, k)]
                                + S->value_s * u[S->index_s(0, j, k)]
                                + S->value_c * u[S->index_c(0, j, k)]
                                + S->value_e * u[S->index_e(0, j, k)]
                                + S->value_n * u[S->index_n(0, j, k)]
                                + S->value_t * u[S->index_t(0, j, k)];
    }
  }

  //face: k=1,...,nz-2, j=1,...,ny-2, i=nx-1
  #pragma omp parallel for schedule(static) collapse(2) //<-for task 6 hw1, uncomment
  for (int k = 1; k < S->nz - 1; k++)
  {
    for (int j = 1; j < S->ny - 1; j++)
    {
        v[S->index_c(S->nx - 1, j, k)]  = S->value_b * u[S->index_b(S->nx - 1, j, k)]
                                        + S->value_s * u[S->index_s(S->nx - 1, j, k)]
                                        + S->value_w * u[S->index_w(S->nx - 1, j, k)]
                                        + S->value_c * u[S->index_c(S->nx - 1, j, k)]
                                        + S->value_n * u[S->index_n(S->nx - 1, j, k)]
                                        + S->value_t * u[S->index_t(S->nx - 1, j, k)];
    }
  }
  
  //lines: i=1,..nx-2
  #pragma omp parallel for schedule(static)
  for (int i = 1; i < S->nx - 1; i++)
  {
    //k=0, j=0
    v[S->index_c(i, 0, 0)]  = S->value_w * u[S->index_w(i, 0, 0)]
                            + S->value_c * u[S->index_c(i, 0, 0)]
                            + S->value_e * u[S->index_e(i, 0, 0)]
                            + S->value_n * u[S->index_n(i, 0, 0)]
                            + S->value_t * u[S->index_t(i, 0, 0)];
    //k=0, j=ny-1
    v[S->index_c(i, S->ny - 1, 0)]  = S->value_s * u[S->index_s(i, S->ny - 1, 0)]
                                    + S->value_w * u[S->index_w(i, S->ny - 1, 0)]
                                    + S->value_c * u[S->index_c(i, S->ny - 1, 0)]
                                    + S->value_e * u[S->index_e(i, S->ny - 1, 0)]
                                    + S->value_t * u[S->index_t(i, S->ny - 1, 0)];
    
    //k=nz-1, j=0
    v[S->index_c(i, 0, S->nz - 1)]  = S->value_b * u[S->index_b(i, 0, S->nz - 1)]
                                    + S->value_w * u[S->index_w(i, 0, S->nz - 1)]
                                    + S->value_c * u[S->index_c(i, 0, S->nz - 1)]
                                    + S->value_e * u[S->index_e(i, 0, S->nz - 1)]
                                    + S->value_n * u[S->index_n(i, 0, S->nz - 1)];
    
    //k=nz-1, j=ny-1
    v[S->index_c(i, S->ny - 1, S->nz - 1)]  = S->value_b * u[S->index_b(i, S->ny - 1, S->nz - 1)]
                                            + S->value_s * u[S->index_s(i, S->ny - 1, S->nz - 1)]
                                            + S->value_w * u[S->index_w(i, S->ny - 1, S->nz - 1)]
                                            + S->value_c * u[S->index_c(i, S->ny - 1, S->nz - 1)]
                                            + S->value_e * u[S->index_e(i, S->ny - 1, S->nz - 1)];
  }
  
  //lines: j=1,..nx-2
  #pragma omp parallel for schedule(static)
  for (int j = 1; j < S->ny - 1; j++)
  {
    //k=0, i=0
    v[S->index_c(0, j, 0)]  = S->value_s * u[S->index_s(0, j, 0)]
                            + S->value_c * u[S->index_c(0, j, 0)]
                            + S->value_e * u[S->index_e(0, j, 0)]
                            + S->value_n * u[S->index_n(0, j, 0)]
                            + S->value_t * u[S->index_t(0, j, 0)];
    
    //k=0, i=ny-1
    v[S->index_c(S->nx - 1, j, 0)]  = S->value_s * u[S->index_s(S->nx - 1, j, 0)]
                                    + S->value_w * u[S->index_w(S->nx - 1, j, 0)]
                                    + S->value_c * u[S->index_c(S->nx - 1, j, 0)]
                                    + S->value_n * u[S->index_n(S->nx - 1, j, 0)]
                                    + S->value_t * u[S->index_t(S->nx - 1, j, 0)];
    
    //k=nz-1, i=0
    v[S->index_c(0, j, S->nz - 1)]  = S->value_b * u[S->index_b(0, j, S->nz - 1)]
                                    + S->value_s * u[S->index_s(0, j, S->nz - 1)]
                                    + S->value_c * u[S->index_c(0, j, S->nz - 1)]
                                    + S->value_e * u[S->index_e(0, j, S->nz - 1)]
                                    + S->value_n * u[S->index_n(0, j, S->nz - 1)];
    
    //k=nz-1, i=ny-1
    v[S->index_c(S->nx - 1, j, S->nz - 1)]  = S->value_b * u[S->index_b(S->nx - 1, j, S->nz - 1)]
                                            + S->value_s * u[S->index_s(S->nx - 1, j, S->nz - 1)]
                                            + S->value_w * u[S->index_w(S->nx - 1, j, S->nz - 1)]
                                            + S->value_c * u[S->index_c(S->nx - 1, j, S->nz - 1)]
                                            + S->value_n * u[S->index_n(S->nx - 1, j, S->nz - 1)];
  }
  
  //lines: k=1,..nx-2
  #pragma omp parallel for schedule(static)
  for (int k = 1; k < S->nz - 1; k++)
  {
    //j=0, i=0
    v[S->index_c(0, 0, k)]  = S->value_b * u[S->index_b(0, 0, k)]
                            + S->value_c * u[S->index_c(0, 0, k)]
                            + S->value_e * u[S->index_e(0, 0, k)]
                            + S->value_n * u[S->index_n(0, 0, k)]
                            + S->value_t * u[S->index_t(0, 0, k)];
    
    //j=0, i=nx-1
    v[S->index_c(S->nx - 1, 0, k)]  = S->value_b * u[S->index_b(S->nx - 1, 0, k)]
                                    + S->value_w * u[S->index_w(S->nx - 1, 0, k)]
                                    + S->value_c * u[S->index_c(S->nx - 1, 0, k)]
                                    + S->value_n * u[S->index_n(S->nx - 1, 0, k)]
                                    + S->value_t * u[S->index_t(S->nx - 1, 0, k)];
    
    //j=ny-1, i=0
    v[S->index_c(0, S->ny - 1, k)]  = S->value_b * u[S->index_b(0, S->ny - 1, k)]
                                    + S->value_s * u[S->index_s(0, S->ny - 1, k)]
                                    + S->value_c * u[S->index_c(0, S->ny - 1, k)]
                                    + S->value_e * u[S->index_e(0, S->ny - 1, k)]
                                    + S->value_t * u[S->index_t(0, S->ny - 1, k)];
    
    //j=ny-1, i=nx-1
    v[S->index_c(S->nx - 1, S->ny - 1, k)]  = S->value_b * u[S->index_b(S->nx - 1, S->ny - 1, k)]
                                            + S->value_s * u[S->index_s(S->nx - 1, S->ny - 1, k)]
                                            + S->value_w * u[S->index_w(S->nx - 1, S->ny - 1, k)]
                                            + S->value_c * u[S->index_c(S->nx - 1, S->ny - 1, k)]
                                            + S->value_t * u[S->index_t(S->nx - 1, S->ny - 1, k)];
    
  }

  //corner: k=0,j=0,i=0
  v[S->index_c(0, 0, 0)]    = S->value_c * u[S->index_c(0, 0, 0)] 
                            + S->value_e * u[S->index_e(0, 0, 0)] 
                            + S->value_n * u[S->index_n(0, 0, 0)] 
                            + S->value_t * u[S->index_t(0, 0, 0)];
      
  //corner: k=0,j=0,i=nx-1
  v[S->index_c(S->nx - 1, 0, 0)]    = S->value_w * u[S->index_w(S->nx - 1, 0, 0)] 
                                    + S->value_c * u[S->index_c(S->nx - 1, 0, 0)] 
                                    + S->value_n * u[S->index_n(S->nx - 1, 0, 0)] 
                                    + S->value_t * u[S->index_t(S->nx - 1, 0, 0)];

 //corner: k=0,j=ny-1,i=0
  v[S->index_c(0, S->ny - 1, 0)]    =  S->value_s * u[S->index_s(0, S->ny - 1, 0)] 
                                    + S->value_c * u[S->index_c(0, S->ny - 1, 0)] 
                                    +  S->value_e * u[S->index_e(0, S->ny - 1, 0)] 
                                    + S->value_t * u[S->index_t(0, S->ny - 1, 0)];
      
  //corner: k=0,j=ny-1,i=nx-1
  v[S->index_c(S->nx - 1, S->ny - 1, 0)]    = S->value_s * u[S->index_s(S->nx - 1, S->ny - 1, 0)] 
                                            + S->value_w * u[S->index_w(S->nx - 1, S->ny - 1, 0)] 
                                            + S->value_c * u[S->index_c(S->nx - 1, S->ny - 1, 0)] 
                                            + S->value_t * u[S->index_t(S->nx - 1, S->ny - 1, 0)];
      
  //corner: k=nz-1,j=0,i=0
  v[S->index_c(0, 0, S->nz - 1)]    = S->value_b * u[S->index_b(0, 0, S->nz - 1)] 
                                    + S->value_c * u[S->index_c(0, 0, S->nz - 1)] 
                                    + S->value_e * u[S->index_e(0, 0, S->nz - 1)] 
                                    + S->value_n * u[S->index_n(0, 0, S->nz - 1)];
      
  //corner: k=nz-1,j=0,i=nx-1
  v[S->index_c(S->nx - 1, 0, S->nz - 1)]    = S->value_b * u[S->index_b(S->nx - 1, 0, S->nz - 1)] 
                                            + S->value_w * u[S->index_w(S->nx - 1, 0, S->nz - 1)]
                                            + S->value_c * u[S->index_c(S->nx - 1, 0, S->nz - 1)] 
                                            + S->value_n * u[S->index_n(S->nx - 1, 0, S->nz - 1)];
      

  //corner: k=nz-1,j=ny-1,i=0
  v[S->index_c(0, S->ny - 1, S->nz - 1)]    = S->value_b * u[S->index_b(0, S->ny - 1, S->nz - 1)] 
                                            + S->value_s * u[S->index_s(0, S->ny - 1, S->nz - 1)] 
                                            + S->value_c * u[S->index_c(0, S->ny - 1, S->nz - 1)] 
                                            + S->value_e * u[S->index_e(0, S->ny - 1, S->nz - 1)];

  //corner: k=nz-1,j=ny-1,i=nx-1
  v[S->index_c(S->nx - 1, S->ny - 1, S->nz - 1)]    = S->value_b * u[S->index_b(S->nx - 1, S->ny - 1, S->nz - 1)] 
                                                    + S->value_s * u[S->index_s(S->nx - 1, S->ny - 1, S->nz - 1)] 
                                                    + S->value_w * u[S->index_w(S->nx - 1, S->ny - 1, S->nz - 1)] 
                                                    + S->value_c * u[S->index_c(S->nx - 1, S->ny - 1, S->nz - 1)];
  
  return;
}

void apply_diagonalMatrix(int n, double s,
        double const* u, double* v)
{
  #pragma omp parallel for schedule(static)
  for (int t = 0; t < n; t++)
    v[t] = s * u[t];
  return;
}