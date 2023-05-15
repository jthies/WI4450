#include "time_integration.hpp"
#include "operations.hpp"

#include <cmath>
#include <stdexcept>
#include <algorithm>

#include <iostream>
#include <iomanip>
#include <omp.h>

void time_integration_parallel_L_parallel_Jacobi(stencil3d const* op, int n, double* x, double const* b,
        double tol, double delta_t, int maxIter, int T,
        double* resNorm, int* numIter,
        int verbose)
{
  if (op->nx * op->ny * op->nz != n)
  {
    throw std::runtime_error("mismatch between stencil and vector dimension passed to cg_solver");
  }
  
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
      std::cout << std::setw(4) << iter + 1 << "\t" << std::setw(8) << std::setprecision(4) << rho << std::endl;
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

void time_integration_gmres(stencil3d const* op, int n, double* x, const double* b, double epsilon, double delta_t, int const maxIter, int T, double* resNorm){
    int const maxIter_p1 = maxIter + 1;
    double *Q = new double[n*T*maxIter_p1];
    double H[maxIter_p1*maxIter] = {0.0};
    double e_1[maxIter_p1] = {0.0};
    double *r = new double[n*T];
    double *Ax = new double[n*T];
    double *AQ = new double[n*T];
    double *Q_j = new double[n*T];
    double r_norm = 1.0;
    double res = 1.0;
    
    // Set e_1 to be [1,0,...0]
    e_1[0] = 1.0;

    // Ax = A*x
    Ax_apply_stencil(op, x, Ax, T, n, delta_t);

    // Ax (= r_0) = b - Ax
    axpby(n*T, 1.0, b, -1.0, Ax);
    r_norm = sqrt(dot(n, Ax, Ax));
    // Q[:][0] = r_0/||r_0||^2
    for(int i=0; i<n*T; i++){
           Q[index(i, 0, maxIter_p1)]= Ax[i]/r_norm;
        }
    // Perform the Arnoldi iteration
    for(int j=0; j<maxIter; j++){
        std::cout << "Iteratie " << j << std::endl;
        // Calculate A*Q[:,j] and put it into Q[:, j+1]
        // Put Q[:,j] into Q_j and Q[:,]
        for (int i=0; i<n*T; i++){
            Q_j[i] = Q[index(i,j,maxIter_p1)];
        }
        Ax_apply_stencil(op, Q_j, AQ, T, n, delta_t);
        // Put AQ into Q[:,j+1]
        for (int i=0; i<n; i++){
            Q[index(i,j+1,maxIter_p1)] = AQ[i];
        }
        
        // TODO misschien twee losse for loops
        for (int i=0; i<j; i++){
            // H[i][j] = Q[:][i]^T*Q[:,j+1]
            for (int k=0;k<n;k++){
                H[index(i,j,maxIter)] += Q[index(k,i,maxIter_p1)]*Q[index(k,j+1,maxIter_p1)];
            }
            // Q[:][j+1] = Q[:][j+1] - H[i][j]*Q[:,i]
            for (int k=0;k<n;k++){
                Q[index(k,j+1,maxIter_p1)] = Q[index(k,j+1,maxIter_p1)] - H[index(i,j,maxIter)]*Q[index(k,i,maxIter_p1)];
            }
        }

        // H[j+1][j] = norm(Q[:][j+1])
        for (int k=0;k<n;k++){
            H[index(j+1,j,maxIter)] = Q[index(k,j+1,maxIter_p1)]*Q[index(k,j+1,maxIter_p1)];
        }
        H[index(j+1,j,maxIter)] = sqrt(H[index(j+1,j,maxIter)]);

        // Avoid dividing by zero
        if (abs(H[index(j+1,j,maxIter)]) > epsilon) {
            // Q[:][j+1] = Q[:][j+1]/H[j+1][j]
            for (int k=0;k<n;k++){
                Q[index(k,j+1,maxIter_p1)] = Q[index(k,j+1,maxIter_p1)]/H[index(j+1,j,maxIter)];
            }
        }
        
        // Solve for y: H[:j+2][:j+1]*y = beta*e_1
        for (int i=0; i<maxIter+1; i++){
            for (int k=0; k<maxIter; k++){
                std::cout << H[index(i,j,maxIter)] << " ";
            }
            std::cout << std::endl;
        }

    // //     y = ...


    // //     res = dot(n,)
    // //     if (res < epsilon){
    // //         for (int i=0;i<n;i++){
    // //             for (int k=0;k<n;k++){
    // //                 sol[i] = Q[i][k]*Y[k] + x0[i]
    // //             }
    // //         }
    // //         return sol, res
    // //     }

    }
    // for (int i=0;i<n;i++){
    //     for (int k=0;k<n;k++){
    //         sol[i] = Q[i][k]*Y[k] + x0[i]
    //     }
    // }
    *resNorm = res;

    delete [] Q;
    delete [] r;
    delete [] Ax;
    delete [] AQ;
    delete [] Q_j;

    return;
}




