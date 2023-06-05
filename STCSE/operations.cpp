#include "operations.hpp"
#include <omp.h>
#include <iostream>

int index(int x, int y, int n_row){
    return x + y*n_row;
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

void vector_scale(int n, const double* scale,double* x){
  #pragma omp parallel for
  for (int i=0; i<n; i++){
    x[i] /= scale[i];
  } 
}

void vec2vec(int n, const double* x, double* y){
  #pragma omp parallel for
  for (int i=0; i<n; i++){
    y[i] = x[i];
  } 
}

void matrix2vec(int n, int const j, double* x, double const* Q){
  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    x[i] = Q[index(i, j, n)];
  }
}

void vec2matrix(int n, int const j,double const* x, double* Q){
  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    Q[index(i, j, n)] = x[i];
  }
}

double matrix_col_vec_dot(int n, int const j, double const* x, double const* Q){
  double dot = 0.0;
  #pragma omp parallel for reduction(+:dot)
  for (int i = 0; i < n; i++) {
    dot += Q[index(i, j, n)] * x[i];
  }
  return dot;
}

double matrix_col_dot(int n, int const j, double const* Q){
  double dot = 0.0;
  #pragma omp parallel for reduction(+:dot)
  for (int i = 0; i < n; i++) {
    dot += Q[index(i, j, n)] * Q[index(i, j, n)];
  }
  return dot;
}

void matrix_col_scale(int n, int const j, double a, double* Q){
  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    Q[index(i, j, n)] = Q[index(i, j, n)] / a;
  }
}

void orthogonalize_Q(int n, int n_H, int const i,int const j,double *Q,double const* H){
  #pragma omp parallel for
  for (int k = 0; k < n ; k++) {
    Q[index(k, j, n)] -= H[index(i, j-1, n_H)] * Q[index(k, i, n)];
  }
}

void matrix_vec_prod(int n_r, int n_col, double* x, const double* Q, const double* y){
    #pragma omp parallel for
    for (int i = 0; i < n_r; i++) {
        double sum = 0.0;
        for (int j = 0; j < n_col; j++) {
            sum += Q[index(i, j, n_r)] * y[j];
        }
        x[i] = sum;
    }
}
void print_vec(int n,const double* x){
  for (int i = 0; i < n; i++) {
    std::cout<<x[i]<<std::endl;
  }
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

void Ax_apply_stencil_forward_euler(const int n, const int T, const double deltaT, const double* x, double* Ax, const stencil3d *op){
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
      // x_k_min_1 = - x_k_min_1 - delta_t*Lx_k_min_1
      axpby(n, -deltaT, Lx_k_min_1, -1.0, x_k_min_1);
    }
    // x_k = x_k + x_k_min_1 = x_k - x_k_min_1 - delta_t*Lx_k_min_1
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

void Ax_apply_stencil_backward_euler(const int n, const int T, const double deltaT, const double* x, double* Ax, const stencil3d *op){
  #pragma omp parallel for
  for (int k=0; k<T; k++){
    double *x_k_min_1 = new double[n];
    double *x_k = new double[n];
    double *Lx_k = new double[n];
    // copy x[k*n to (k+1)*n] into x_k
    for(int l=k*n; l<(k+1)*n; l++){
        x_k[l-k*n] = x[l];
      }
    // Initialize the previous vector with zeros for the first time step
    if (k==0){
      init(n, x_k_min_1, 0.0);
    } 
    // Copy the previous timestep solution into x_k_min_1
    else {
      // copy x[(k-1)*n to k*n] into x_k_min_1
      for(int l=(k-1)*n; l<k*n; l++){
        x_k_min_1[l-(k-1)*n] = x[l];
      }
      // Lx_k = op*x_k (L is the operator)
      apply_stencil3d(op, x_k, Lx_k);
      // x_k = x_k - delta_t*Lx_k
      axpby(n, -deltaT, Lx_k, 1.0, x_k);
    }
    // x_k = x_k - x_k_min_1 = x_k - delta_t*Lx_k - x_k_min_1
    axpby(n, -1.0, x_k_min_1, 1.0, x_k);

    // Copy x_k into Ax
    for(int l=k*n; l<(k+1)*n; l++){
        Ax[l] = x_k[l-k*n]; 
    }
    delete [] x_k_min_1;
    delete [] x_k;
    delete [] Lx_k;
  }
  return;
}


