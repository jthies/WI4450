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

void apply_givens_rotation(double& a, double& b, double& c, double& s) {
  if (b == 0.0) {
    c = 1.0;
    s = 0.0;
  } 
  else {
    if (abs(b) > abs(a)) {
      double tau = -a / b;
      s = 1.0 / sqrt(1.0 + tau * tau);
      c = s * tau;
    } else {
      double tau = -b / a;
      c = 1.0 / sqrt(1.0 + tau * tau);
      s = c * tau;
    }
  }
  // Apply rotation to a and b
  double temp = c * a - s * b;
  b = s * a + c * b;
  a = temp;
}

void apply_givens_to_H_e1(double* H, double* e_1, double c, double s, int j, int j1, int maxIter_p1) {
    for (int i = 0; i < maxIter_p1; ++i) {
        double tempH = c * H[index(j, i, maxIter_p1)] - s * H[index(j1, i, maxIter_p1)];
        H[index(j1, i, maxIter_p1)] = s * H[index(j, i, maxIter_p1)] + c * H[index(j1, i, maxIter_p1)];
        H[index(j, i, maxIter_p1)] = tempH;
    }

    double tempe_1 = c * e_1[j] - s * e_1[j1];
    e_1[j1] = s * e_1[j] + c * e_1[j];
    e_1[j] = tempe_1;
}




void time_integration_gmres(stencil3d const* op, int n, double* x, const double* b, double epsilon, double delta_t, int const maxIter, int T, double* resNorm){
    int const maxIter_p1 = maxIter + 1;
    double Q[n*T*maxIter_p1] = {0.0};
    double H[maxIter_p1*maxIter] = {0.0};
    double H_g[maxIter_p1*maxIter] = {0.0};
    double *Ry = new double[maxIter_p1];
    double *e_1 = new double[maxIter_p1];
    double *e_1_g = new double[maxIter_p1];
    double *sol = new double[n*T];
    double *r = new double[n*T];
    double *Ax = new double[n*T];
    double *AQ = new double[n*T];
    double *Q_j = new double[n*T];
    double r_norm = 1.0;
    double res = 1.0;
    double denom = 0.0;
    double *y = new double[maxIter_p1](); // initialize y to 0
    double c = 0.0;
    double s = 0.0;

    
    // Ax = A*x
    Ax_apply_stencil(op, x, Ax, T, n, delta_t);
    // Ax (= r_0) = b - Ax
    axpby(n*T, 1.0, b, -1.0, Ax);
    r_norm = sqrt(dot(n*T, Ax, Ax));
    // Set e_1 to be [beta,0,...0]
    init(maxIter_p1, e_1, 0.0);
    e_1[0] = r_norm;

    // Q[:][0] = r_0/||r_0||^2
    for(int i=0; i<n*T; i++){
      Q[index(i, 0, n*T)]= Ax[i]/r_norm;
    }
    // Perform the Arnoldi iteration
    for(int j=0; j<maxIter; j++){
      std::cout << "Iteratie " << j << std::endl;
      // Calculate A*Q[:,j] and put it into Q[:, j+1]
      // Put Q[:,j] into Q_j
      for (int i=0; i<n*T; i++){
        Q_j[i] = Q[index(i,j,n*T)];
      }

      Ax_apply_stencil(op, Q_j, AQ, T, n, delta_t);

      // Put AQ into Q[:,j+1]
      for (int i=0; i<n*T; i++){
        Q[index(i,j+1,n*T)] = AQ[i];
      }

      for (int i=0; i<=j; i++){
        // H[i][j] = Q[:][i]^T*Q[:,j+1]
        for (int k=0; k<n*T; k++){
          // H[index(i,j,maxIter_p1)] += Q[index(k,i,n*T)]*Q[index(k,j+1,n*T)];
          H[index(i,j,maxIter_p1)] += Q[index(k,i,n*T)]*AQ[k];
        }
        
        // Q[:][j+1] = Q[:][j+1] - H[i][j]*Q[:,i]
        for (int k=0; k<n*T; k++){
          Q[index(k,j+1,n*T)] = Q[index(k,j+1,n*T)] - H[index(i,j,maxIter_p1)]*Q[index(k,i,n*T)];
        }
      }

      // H[j+1][j] = norm(Q[:][j+1])
      H[index(j+1,j,maxIter_p1)] = 0.0;
      for (int k=0;k<n*T;k++){
        H[index(j+1,j,maxIter_p1)] += Q[index(k,j+1,n*T)]*Q[index(k,j+1,n*T)];
      }
      H[index(j+1,j,maxIter_p1)] = sqrt(H[index(j+1,j,maxIter_p1)]);
      

      // Q[:][j+1] = Q[:][j+1]/H[j+1][j]
      for (int k=0;k<n*T;k++){
        Q[index(k,j+1,n*T)] = Q[index(k,j+1,n*T)]/H[index(j+1,j,maxIter_p1)];
      }
      
      // Check if Q_i with Q_j+1 are orthonormal
      for (int i=0; i<=j+1; i++){
        double nor = 0.0;
        for (int k=0;k<n*T;k++){
          nor += Q[index(k,i,n*T)]*Q[index(k,j+1,n*T)];
        }
        std::cout << "norm" << nor << " ";
      }

      std::cout << "H and e_1 without Givens rotation" << std::endl;
      for (int i=0; i<maxIter_p1; i++){
          for (int k=0; k<maxIter; k++){
              std::cout << H[index(i,k,maxIter_p1)] << " ";
          }
          std::cout << std::endl;
      }

      std::cout << "e_1 ";
      for (int i = 0; i < j+1; i++){
        std::cout << e_1[i] << " ";
      }
      std::cout << std::endl;

    //  // Givens rotation on H_:j+2,:j+1 to make upper triangular matrix = R
    //   denom = sqrt(H[index(j,j,maxIter_p1)]*H[index(j,j,maxIter_p1)] + H[index(j+1,j,maxIter)]*H[index(j+1,j,maxIter)]);
    //   c = H[index(j,j,maxIter)]/ denom;
    //   s = - H[index(j+1,j,maxIter)]/ denom;
    //   for (int i = 0; i< maxIter; i++){
    //     H_g[index(j,i,maxIter)] = c*H[index(j,i,maxIter)] + s*H[index(j+1,i,maxIter)];
    //     H_g[index(j+1,i,maxIter)] =  -s*H[index(j,i,maxIter)] + c*H[index(j+1,i,maxIter)];
    //   }
    //   e_1_g[j] = c*e_1[j]+ s*e_1[j+1];
    //   e_1_g[j+1] = -s*e_1[j] + c*e_1[j+1];
    //   e_1[j] = e_1_g[j];
    //   e_1[j+1] = e_1_g[j+1];
    //   for (int i = 0; i< maxIter; i++){
    //     H[index(j,i,maxIter)] = H_g[index(j,i,maxIter)];
    //     H[index(j+1,i,maxIter)] = H_g[index(j+1,i,maxIter)];
    //   }

    //   std::cout << "H with Givens rotation" << std::endl;
    //   for (int i=0; i<maxIter+1; i++){
    //       for (int k=0; k<maxIter; k++){
    //           std::cout << H[index(i,k,maxIter)] << " ";
    //       }
    //       std::cout << std::endl;
    //   }

    //   std::cout << "e_1 ";
    //   for (int i = 0; i < maxIter; i++){
    //     std::cout << e_1[i] << " ";
    //   }
    //   std::cout << std::endl;

    //   // Calculate y using backward substitution
    //   for (int i = j+1-1; i >= 0; i--){
    //     y[i] = e_1[i];
    //     for (int k = i + 1; k < j+2-1; k++){
    //         y[i] -= H[index(i,k,maxIter)] * y[k];
    //     }
    //     y[i] /= H[index(i,i,maxIter)];
    //   }
    //   std::cout << "y minnorm sol ";
    //   for (int i = 0; i < j+1; i++){
    //     std::cout << y[i] << " ";
    //   }
    //   std::cout << std::endl;

    //   for (int i = 0; i < j+2-1; i++){
    //     for (int k = 0; k < j+1-1; k++){
    //       Ry[i] += H[index(i,k,maxIter)]*y[k];
    //     }
    //   }
    //   axpby(j+2-1, 1.0, e_1, -1.0, Ry);
    //   res = sqrt(dot(j+2-1, Ry, Ry));
    //   if (res < epsilon){
    //     for (int i = 0; i < n*T; i++){
    //       for (int k = 0; k < j+1-1; k++){
    //         sol[i] += Q[index(i,k,maxIter)]*y[k];
    //       }
    //     }
    //     axpby(n*T,1.0,x,1.0,sol);
    //     break;
    //   } else{
    //     std::cout << "residual " << res << std::endl;
    //   }


    }

    *resNorm = res;
    delete [] r;
    delete [] Ax;
    delete [] AQ;
    delete [] Q_j;

    return;
}




