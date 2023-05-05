
#include "time_integration.hpp"
#include "operations.hpp"

#include <cmath>
#include <stdexcept>
#include <algorithm>

#include <iostream>
#include <iomanip>
#include <omp.h>

void time_integration_parallel_L_parallel_Jacobi(stencil3d const* op, int n, double* x, double const* x_0,
        double tol, double delta_t, int maxIter, int T,
        double* resNorm, int* numIter,
        int verbose)
{
  if (op->nx * op->ny * op->nz != n)
  {
    throw std::runtime_error("mismatch between stencil and vector dimension passed to cg_solver");
  }
  
  
  // Create b from the initial vector x_0=u_b and a vector of zeros b_1_to_T
  double *b = new double[n*T];
  double *b_1_to_T = new double[n*(T-1)];
  init(n*(T-1), b_1_to_T, 0.0);
  std::copy(x_0, x_0 + n, b);
  std::copy(b_1_to_T, b_1_to_T + n*(T-1), b + n);
  delete [] b_1_to_T;

  // Create other variables needed
  double *r = new double[n*T];
  double *Ax = new double[n*T];
  
  double rho=1.0;

  // Start by initializing x with the rhs b
  axpby(n*T, 1.0, b, 0.0, x);
  // Loop over number of solver iterations
  for (int iter=0; iter<maxIter; iter++){
    // Loop over number of timesteps 
    // Calculate Ax
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
        apply_stencil3d_parallel(op, x_k_min_1, Lx_k_min_1);
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

    // r = b - Ax 
    axpby(n*T, 1.0, b, 0.0, r); // first copy b to r (we need to keep storing b)
    axpby(n*T, -1.0, Ax, 1.0, r);


    // Calculate norm of r and print it
    rho = dot(n*T,r,r);

    if (verbose)
    {
      std::cout << std::setw(4) << iter << "\t" << std::setw(8) << std::setprecision(4) << rho << std::endl;
      // std::cout << std::setprecision(4) << rho << ","; // if you want to plot it in python
    }

    // x = x + r
    axpby(n*T, 1.0, r, 1.0, x);

  }

  delete [] b;
  delete [] r;
  delete [] Ax;

  *resNorm = rho;

  return;
}

void time_integration_parallel_L_Jacobi(stencil3d const* op, int n, double* x, double const* x_0,
        double tol, double delta_t, int maxIter, int T,
        double* resNorm, int* numIter,
        int verbose)
{
  if (op->nx * op->ny * op->nz != n)
  {
    throw std::runtime_error("mismatch between stencil and vector dimension passed to cg_solver");
  }
  
  
  // Create b from the initial vector x_0=u_b and a vector of zeros b_1_to_T
  double *b = new double[n*T];
  double *b_1_to_T = new double[n*(T-1)];
  init(n*(T-1), b_1_to_T, 0.0);
  std::copy(x_0, x_0 + n, b);
  std::copy(b_1_to_T, b_1_to_T + n*(T-1), b + n);
  delete [] b_1_to_T;

  // Create other variables needed
  double *r = new double[n*T];
  double *Ax = new double[n*T];
  
  double rho=1.0;

  // Start by initializing x with the rhs b
  axpby(n*T, 1.0, b, 0.0, x);
  // Loop over number of solver iterations
  for (int iter=0; iter<maxIter; iter++){
    // Loop over number of timesteps 
    // Calculate Ax
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
        apply_stencil3d_parallel(op, x_k_min_1, Lx_k_min_1);
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

    // r = b - Ax 
    axpby(n*T, 1.0, b, 0.0, r); // first copy b to r (we need to keep storing b)
    axpby(n*T, -1.0, Ax, 1.0, r);


    // Calculate norm of r and print it
    rho = dot(n*T,r,r);

    if (verbose)
    {
      std::cout << std::setw(4) << iter << "\t" << std::setw(8) << std::setprecision(4) << rho << std::endl;
      // std::cout << std::setprecision(4) << rho << ","; // if you want to plot it in python
    }

    // x = x + r
    axpby(n*T, 1.0, r, 1.0, x);

  }

  delete [] b;
  delete [] r;
  delete [] Ax;

  *resNorm = rho;

  return;
}

void time_integration_parallel_Jacobi(stencil3d const* op, int n, double* x, double const* x_0,
        double tol, double delta_t, int maxIter, int T,
        double* resNorm, int* numIter,
        int verbose)
{
  if (op->nx * op->ny * op->nz != n)
  {
    throw std::runtime_error("mismatch between stencil and vector dimension passed to cg_solver");
  }
  
  
  // Create b from the initial vector x_0=u_b and a vector of zeros b_1_to_T
  double *b = new double[n*T];
  double *b_1_to_T = new double[n*(T-1)];
  init(n*(T-1), b_1_to_T, 0.0);
  std::copy(x_0, x_0 + n, b);
  std::copy(b_1_to_T, b_1_to_T + n*(T-1), b + n);
  delete [] b_1_to_T;

  // Create other variables needed
  double *r = new double[n*T];
  double *Ax = new double[n*T];
  
  double rho=1.0;

  // Start by initializing x with the rhs b
  axpby(n*T, 1.0, b, 0.0, x);
  // Loop over number of solver iterations
  for (int iter=0; iter<maxIter; iter++){
    // Loop over number of timesteps 
    // Calculate Ax
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

    // r = b - Ax 
    axpby(n*T, 1.0, b, 0.0, r); // first copy b to r (we need to keep storing b)
    axpby(n*T, -1.0, Ax, 1.0, r);


    // Calculate norm of r and print it
    rho = dot(n*T,r,r);

    if (verbose)
    {
      std::cout << std::setw(4) << iter << "\t" << std::setw(8) << std::setprecision(4) << rho << std::endl;
      // std::cout << std::setprecision(4) << rho << ","; // if you want to plot it in python
    }

    // x = x + r
    axpby(n*T, 1.0, r, 1.0, x);

  }

  delete [] b;
  delete [] r;
  delete [] Ax;

  *resNorm = rho;

  return;
}

void time_integration_Jacobi(stencil3d const* op, int n, double* x, double const* x_0,
        double tol, double delta_t, int maxIter, int T,
        double* resNorm, int* numIter,
        int verbose)
{
  if (op->nx * op->ny * op->nz != n)
  {
    throw std::runtime_error("mismatch between stencil and vector dimension passed to cg_solver");
  }
  
  
  // Create b from the initial vector x_0=u_b and a vector of zeros b_1_to_T
  double *b = new double[n*T];
  double *b_1_to_T = new double[n*(T-1)];
  init(n*(T-1), b_1_to_T, 0.0);
  std::copy(x_0, x_0 + n, b);
  std::copy(b_1_to_T, b_1_to_T + n*(T-1), b + n);
  delete [] b_1_to_T;

  // Create other variables needed
  double *r = new double[n*T];
  double *Ax = new double[n*T];
  
  double rho=1.0;

  // Start by initializing x with the rhs b
  axpby(n*T, 1.0, b, 0.0, x);
  // Loop over number of solver iterations
  for (int iter=0; iter<maxIter; iter++){
    // Loop over number of timesteps 
    // Calculate Ax
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

    // r = b - Ax 
    axpby(n*T, 1.0, b, 0.0, r); // first copy b to r (we need to keep storing b)
    axpby(n*T, -1.0, Ax, 1.0, r);


    // Calculate norm of r and print it
    rho = dot(n*T,r,r);

    if (verbose)
    {
      std::cout << std::setw(4) << iter << "\t" << std::setw(8) << std::setprecision(4) << rho << std::endl;
      // std::cout << std::setprecision(4) << rho << ","; // if you want to plot it in python
    }

    // x = x + r
    axpby(n*T, 1.0, r, 1.0, x);

  }

  delete [] b;
  delete [] r;
  delete [] Ax;

  *resNorm = rho;

  return;
}



void time_integration_GMRES(stencil3d const* op, int n, double* x, double const* x_0,
        double tol, double delta_t, int maxIter, int T,
        double* resNorm, int* numIter,
        int verbose)
{
  if (op->nx * op->ny * op->nz != n)
  {
    throw std::runtime_error("mismatch between stencil and vector dimension passed to cg_solver");
  }
  
  
  // Create b from the initial vector x_0=u_b and a vector of zeros b_1_to_T
  double *b = new double[n*T];
  double *b_1_to_T = new double[n*(T-1)];
  init(n*(T-1), b_1_to_T, 0.0);
  std::copy(x_0, x_0 + n, b);
  std::copy(b_1_to_T, b_1_to_T + n*(T-1), b + n);
  delete [] b_1_to_T;

  // Create other variables needed
  double *r = new double[n*T];
  double *Ax = new double[n*T];
  
  double rho=1.0;

  // Start by initializing x with the rhs b
  axpby(n*T, 1.0, b, 0.0, x);
  // Loop over number of solver iterations
  for (int iter=0; iter<maxIter; iter++){
    // Loop over number of timesteps 
    // Calculate Ax
    


    
    
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

    // r = b - Ax 
    axpby(n*T, 1.0, b, 0.0, r); // first copy b to r (we need to keep storing b)
    axpby(n*T, -1.0, Ax, 1.0, r);


    // Calculate norm of r and print it
    rho = dot(n*T,r,r);

    if (verbose)
    {
      std::cout << std::setw(4) << iter << "\t" << std::setw(8) << std::setprecision(4) << rho << std::endl;
      // std::cout << std::setprecision(4) << rho << ","; // if you want to plot it in python
    }

    // x = x + r
    axpby(n*T, 1.0, r, 1.0, x);

  }

  delete [] b;
  delete [] r;
  delete [] Ax;

  *resNorm = rho;

  return;
}





