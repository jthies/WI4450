#include "time_integration.hpp"
#include "operations.hpp"
#include "cg_solver.hpp"

#include <cmath>
#include <stdexcept>
#include <algorithm>

#include <iostream>
#include <iomanip>
#include <omp.h>

void time_integration_sequential_FE(stencil3d const* op, int n, double* x, double const* x_0,
        double  tol, double delta_t,   int  maxIter, int T,
        double* resNorm, int* numIter,
        int verbose)
{
  if (op->nx * op->ny * op->nz != n)
  {
    throw std::runtime_error("mismatch between stencil and vector dimension passed to time integration");
  }
  
  // Create other variables needed
  double *Ax = new double[n*T];
  double *r = new double[n*T];
  double *x_old = new double[n];
  double *x_new = new double[n];

  // Set the first x_old to be x_0 and put it in the solution vector x
  axpby(n, 1.0, x_0, 0.0, x_old);
  for (int k=0; k < n; k++){
    x[k] = x_0[k];
  }

  for (int i=0; i < T-1; i++){ //in iteration T-2, x_{T-1} is calculated
    // x_new = x_old + delta_t*L*x_old
    apply_stencil3d_parallel(op, x_old, x_new);
    axpby(n, 1.0, x_old, delta_t, x_new);
    
    // Copy x_new into the solution vector
    for (int k=0; k < n; k++){
      x[k+(i+1)*n] = x_new[k];
    }

    // x_old = x_new
    axpby(n, 1.0, x_new, 0.0, x_old);
  }

  Ax_apply_stencil(op, x, Ax, T, n, delta_t);
  // r = b - Ax 
  axpby(n*T, 1.0, x_0, 0.0, r); // first copy b to r (we need to keep storing b)
  axpby(n*T, -1.0, Ax, 1.0, r);

  // Calculate norm of r and print it
  *resNorm = dot(n*T,r,r);

  delete [] Ax;
  delete [] r;
  delete [] x_old;
  delete [] x_new;
}


void time_integration_sequential_BE(stencil3d const* op, int n, double* x, double const* x_0,
        double  tol, double delta_t,   int  maxIter, int T,
        double* resNorm, int* numIter,
        int verbose)
{
  if (op->nx * op->ny * op->nz != n)
  {
    throw std::runtime_error("mismatch between stencil and vector dimension passed to time integration");
  }
  
  // Create other variables needed
  double *x_old = new double[n];
  double *x_new = new double[n];

  // Set the first x_old to be x_0 and put it in the solution vector x
  axpby(n, 1.0, x_0, 0.0, x_old);
  for (int k=0; k < n; k++){
    x[k] = x_0[k];
  }

  // make the stencil I-delta_t*L
  stencil3d IdtL;
  IdtL.nx=op->nx; IdtL.ny=op->ny; IdtL.nz=op->nz;
  IdtL.value_c = 1.0 - delta_t * op->value_c;
  IdtL.value_n = -delta_t*op->value_n;
  IdtL.value_e = -delta_t*op->value_e;
  IdtL.value_s = -delta_t*op->value_s;
  IdtL.value_w = -delta_t*op->value_w;
  IdtL.value_t = -delta_t*op->value_t;
  IdtL.value_b = -delta_t*op->value_b;

  for (int i=0; i < T-1; i++){ //in iteration T-2, x_{T-1} is calculated
    
    // solve (I-delta_t*L)x_new = x_old with CG
    cg_solver(&IdtL, n, x_new, x_old, tol, 1e3, resNorm, numIter, 0.0);
    
    // Copy x_new into the solution vector
    for (int k=0; k < n; k++){
      x[k+(i+1)*n] = x_new[k];
    }

    // x_old = x_new
    axpby(n, 1.0, x_new, 0.0, x_old);
  }

  // TODO: calculate norm of error
  *resNorm = 1.0;

  delete [] x_old;
  delete [] x_new;
}

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

void time_integration_gmres(stencil3d const* L, int n, double* x0, const double* b, double epsilon, double delta_t, int maxIter, int T, double* resNorm, int* numIter) {
    int const maxIter_p1 = maxIter + 1;
    double Q[n * T * maxIter_p1] = {0.0};
    double H[maxIter_p1 * maxIter] = {0.0};
    double H_g[maxIter_p1 * maxIter] = {0.0};
    double e_1[maxIter_p1] = {0.0};
    double e_1_g[maxIter_p1] = {0.0};
    double* sol = new double[n * T]();
    double* Asol = new double[n * T]();
    double* r = new double[n * T]();
    double* Ax = new double[n * T]();
    double* AQ = new double[n * T]();
    double* Q_j = new double[n * T]();
    double r_norm = 1.0;
    double b_norm;
    double res = 0.0;
    double denom = 0.0;
    double* y = new double[n * T](); // initialize y to 0
    double* sn = new double[maxIter]; // used in given rotation
    double* cs = new double[maxIter]; // used in given rotation
    int iter;
    srand(1);
    double pertubation = 0;

    init(maxIter, sn, 0.0); 
    init(maxIter, cs, 0.0);


  // Ax = A*x0 --> NOTE: with x0 = 0, Ax = 0 and this computation is not necessary 
  Ax_apply_stencil(L, x0, Ax, T, n, delta_t);
    // apply_stencil3d(L,x0,Ax);
  
  // r_0 (=: Ax) = b - Ax
  axpby(n * T, 1.0, b, -1.0, Ax);
  r_norm = sqrt(dot(n * T, Ax, Ax));
  b_norm = sqrt(dot(n * T, b, b));

  // Q[:][0] = r_0/||r_0||_2
  vec2matrix(n*T,0,Ax, Q);
  matrix_col_scale(n*T,0,r_norm, Q);

  // Set e_1 to be [beta,0,...0]
  init(maxIter_p1, e_1, 0.0);
  e_1[0] = r_norm;
  
  // Perform the Arnoldi iteration
  for (int j = 0; j < maxIter; j++) {
    // Put Q[:,j] into Q_j
    matrix2vec(n*T,j,Q_j,Q);
    // Calculate A*Q[:,j]
    Ax_apply_stencil(L, Q_j, AQ, T, n, delta_t);
    //apply_stencil3d(L,Q_j,AQ);
    // Put AQ into Q[:,j+1]
    vec2matrix(n*T,j+1,AQ,Q);
    for (int i = 0; i < j + 1; i++) {
        // H[i][j] = Q[:][i]^T*Q[:,j+1]
        H[index(i, j, maxIter_p1)] = matrix_col_vec_dot(n*T, i, AQ, Q);
        // Q[:][j+1] = Q[:][j+1] - H[i][j]*Q[:,i]
        orthogonalize_Q(n*T,maxIter_p1,i,j,Q,H);
        // for (int k = 0; k < n * T; k++) {
        //     Q[index(k, j + 1, n * T)] -= H[index(i, j, maxIter_p1)] * Q[index(k, i, n * T)];
        // }
    }
    // H[j+1][j] = ||Q[:][j+1]||_2
    H[index(j + 1, j, maxIter_p1)] = sqrt(matrix_col_dot(n*T,j+1,Q));
    if (H[index(j + 1, j, maxIter_p1)] != 0){
        // Q[:][j+1] = Q[:][j+1]/H[j+1][j]
        matrix_col_scale(n*T,j+1,H[index(j + 1, j, maxIter_p1)],Q);
    } 
    else {
        std::cout << "H[j+1,j] = 0" << std::endl;
    }

    // Remember the e_1 and H as they are original/
    double H_origin[maxIter_p1 * maxIter] = {0.0};
    double e_1_origin[maxIter_p1] = {0.0}; 
    for(int l = 0; l < maxIter_p1 * maxIter; l++){
        H_origin[l] = H[l];
    }
    e_1_origin[0] = r_norm;

    // Givens rotation on H_:j+2,:j+1 to make upper triangular matrix = R        
    given_rotation(j, H, cs, sn, maxIter_p1);

    e_1[j + 1] = -sn[j] * e_1[j];
    e_1[j] = cs[j] * e_1[j];

    // //Print e_1 with the Givens rotation
    // std::cout << "e_1 with Givens rotation" << std::endl;
    // for (int i = 0; i < maxIter; i++) {
    //     std::cout << e_1[i] << " ";
    // }

    // // //Print H with the Givens rotation
    // std::cout << "H with Givens rotation" << std::endl;
    // for (int i = 0; i < maxIter_p1; i++) {
    //     for (int k = 0; k < maxIter; k++) {
    //         std::cout << H[index(i, k, maxIter_p1)] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    std::cout << std::setw(4) << j+1 << "\t" << std::setw(8) << std::setprecision(4) << std::abs(e_1[j+1])/r_norm << std::endl;

    if ((std::abs(e_1[j+1])/r_norm < epsilon)){
        iter = j;
        std::cout <<"GMRES stopped, relative residual < epsilon"<< std::endl;
        break;
    }
    if (j==maxIter-1){
        iter = j;
        std::cout <<"GMRES stopped, max number of iterations is exceeded"<< std::endl;
    }
  } 

  // Back substitution
  init(n*T, y, 0);
  y[iter] = e_1[iter]/H[index(iter, iter, maxIter_p1)];
  for (int i=(iter-1); i>=0; i--){
      y[i] = e_1[i];        
      for (int j=i+1; j <= iter; j++){
          y[i] -= H[index(i, j, maxIter_p1)]*y[j];
      }           
      y[i] = y[i] / H[index(i, i, maxIter_p1)];
  }


  // Actual solution Qy calculation with Q
  init(n * T, sol, 0.0);
  matrix_vec_prod(n*T, iter+2, sol, Q, y);
  // sol = Qy + x0
  axpby(n * T, 1.0, x0, 1.0, sol);

  // Calculate residual b - A*sol
  init(n * T, Asol, 0);
  Ax_apply_stencil(L, sol, Asol, T, n, delta_t);
    // apply_stencil3d(L,sol,Asol);
  axpby(n*T, 1.0, b, -1.0, Asol);

  // Calculate residual norm
  res = std::sqrt(dot(n*T, Asol, Asol));
  // res = sqrt(res)/r_norm;
  std::cout << "relative residual (from b-Ax):"<< res/r_norm <<std::endl;
  std::cout << "relative residual (from e_1[j+1]):"<< std::abs(e_1[iter+1])/r_norm <<std::endl;
  *resNorm = res;
  *numIter = iter;

  delete [] sol;
  delete [] Asol;
  delete [] r;
  delete [] Ax;
  delete [] AQ;
  delete [] Q_j;
}