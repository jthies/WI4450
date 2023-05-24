#include "operations.hpp"
#include <omp.h>

int index(int x, int y, int no_rows){
    return x + y*no_rows;
}

void init(int n, double* x, double value)
{
  #pragma omp parallel for
  for (int i=0; i<n; i++){
    x[i]=value;
  }

  return;
}

double dot(int n, double const* x, double const* y)
{
  double dot_product = 0.0;
  #pragma omp parallel for reduction(+: dot_product)
  for (int i=0; i<n; i++){
    dot_product += x[i]*y[i];
  }

  return dot_product;
}

void axpby(int n, double a, double const* x, double b, double* y)
{
  #pragma omp parallel for
  for (int i=0; i<n; i++){
    y[i] = a*x[i] + b*y[i];
  }

  return;
}

//! apply a 7-point stencil to a vector
void apply_stencil3d_parallel(stencil3d const* S,
        double const* u, double* v)
{
  init((S->nx)*(S->ny)*(S->nz), v, 0.0);

  #pragma omp parallel for collapse(3)
  for (int k=0; k<S->nz; k++){
    for (int j=0; j<S->ny; j++){
        for (int i=0; i<S->nx; i++){
        
        v[S->index_c(i,j,k)] = S->value_c * u[S->index_c(i,j,k)];

        if (i != 0)       {v[S->index_c(i,j,k)] += S->value_w * u[S->index_w(i,j,k)];}
        if (i != S->nx-1) {v[S->index_c(i,j,k)] += S->value_e * u[S->index_e(i,j,k)];}
        if (j != 0)       {v[S->index_c(i,j,k)] += S->value_s * u[S->index_s(i,j,k)];}
        if (j != S->ny-1) {v[S->index_c(i,j,k)] += S->value_n * u[S->index_n(i,j,k)];}
        if (k != 0)       {v[S->index_c(i,j,k)] += S->value_b * u[S->index_b(i,j,k)];}
        if (k != S->nz-1) {v[S->index_c(i,j,k)] += S->value_t * u[S->index_t(i,j,k)];} 
        
      }
    }
  }

  return;
}

void apply_stencil3d(stencil3d const* S,
        double const* u, double* v)
{
  init((S->nx)*(S->ny)*(S->nz), v, 0.0);

  for (int k=0; k<S->nz; k++){
    for (int j=0; j<S->ny; j++){
        for (int i=0; i<S->nx; i++){
        
        v[S->index_c(i,j,k)] = S->value_c * u[S->index_c(i,j,k)];

        if (i != 0)       {v[S->index_c(i,j,k)] += S->value_w * u[S->index_w(i,j,k)];}
        if (i != S->nx-1) {v[S->index_c(i,j,k)] += S->value_e * u[S->index_e(i,j,k)];}
        if (j != 0)       {v[S->index_c(i,j,k)] += S->value_s * u[S->index_s(i,j,k)];}
        if (j != S->ny-1) {v[S->index_c(i,j,k)] += S->value_n * u[S->index_n(i,j,k)];}
        if (k != 0)       {v[S->index_c(i,j,k)] += S->value_b * u[S->index_b(i,j,k)];}
        if (k != S->nz-1) {v[S->index_c(i,j,k)] += S->value_t * u[S->index_t(i,j,k)];} 
        
      }
    }
  }

  return;
}

void Ax_apply_stencil(const stencil3d *op, const double *x, double *Ax, int T, int n, double delta_t){
  #pragma omp parallel for
  for (int k=0; k<T; k++){
    double *x_k_min_1 = new double[n];
    double *x_k = new double[n];
    double *Lx_k_min_1 = new double[n];
    // copy x[k*n to (k+1)*n] into x_k
    for(int l=k*n; l<(k+1)*n; l++){
        x_k[l-k*n] = x[l];
      }
    // Initialize the previous vector with zeros for the first time step
    if (k==0){
      init(n, x_k_min_1, 0);
    } 
    // Copy the previous timestep solution into x_k_min_1
    else {
      // copy x[(k-1)*n to k*n] into x_k_min_1
      for(int l=(k-1)*n; l<k*n; l++){
        x_k_min_1[l-(k-1)*n] = x[l];
      }
      // Lx_k_min_1 = op*x_k_min_1 (L is the operator)
      apply_stencil3d(op, x_k_min_1, Lx_k_min_1);
      // x_k_min_1 = - x_k_min_1 + delta_t*Lx_k_min_1
      axpby(n, delta_t, Lx_k_min_1, -1.0, x_k_min_1);
    }
    // x_k = x_k + x_k_min_1 = x_k - x_k_min_1 + delta_t*Lx_k_min_1
    axpby(n, 1.0, x_k_min_1, 1.0, x_k);

    // Copy x_k into Ax
    for(int l=k*n; l<(k+1)*n; l++){
        Ax[l] = x_k[l-k*n];
    }
    delete [] x_k_min_1;
    delete [] x_k;
    delete [] Lx_k_min_1;
  }
  return;
}

