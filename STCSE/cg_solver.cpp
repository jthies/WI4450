#include "cg_solver.hpp"
#include "operations.hpp"

#include <cmath>
#include <stdexcept>

#include <iostream>
#include <iomanip>

void cg_solver(stencil3d const* op, int n, double* x, double const* b,
        double tol, int maxIter,
        double* resNorm, int* numIter,
        int verbose)
{
  if (op->nx * op->ny * op->nz != n)
  {
    std::cout<<n<<std::endl;
    std::cout<<op->nx<<std::endl;
    std::cout<<op->ny<<std::endl;
    std::cout<<op->nz<<std::endl;
    throw std::runtime_error("mismatch between stencil and vector dimension passed to cg_solver");
  }

  double *p = new double[n];
  double *q = new double[n];
  double *r = new double[n];

  double alpha, beta, rho=1.0, rho_old=0.0;

  // r = op * x
  apply_stencil3d_parallel(op, x, r);
  
  // r = b - r;
  axpby(n, 1.0, b, -1.0, r);

  // p = q = 0
  init(n, p, 0.0);
  init(n, q, 0.0);

  // start CG iteration
  int iter = -1;
  while (true)
  {
    iter++;

    // rho = <r, r>
    rho = dot(n,r,r);

    if (verbose)
    {
      std::cout << std::setw(4) << iter << "\t" << std::setw(8) << std::setprecision(4) << rho << std::endl;
      // std::cout << std::setprecision(4) << rho << ","; // if you want to plot it in python
    }

    // check for convergence or failure
    if ((std::sqrt(rho) < tol) || (iter > maxIter))
    {
      break;
    }

    if (rho_old==0.0)
    {
      alpha = 0.0;
    }
    else
    {
      alpha = rho / rho_old;
    }
    // p = r + alpha * p
    axpby(n, 1.0, r, alpha, p);

    // q = op * p
    apply_stencil3d_parallel(op, p, q);
    
    // beta = <p,q>
    beta = dot(n, p, q);

    alpha = rho / beta;

    // x = x + alpha * p
    axpby(n, alpha, p, 1.0, x);

    // r = r - alpha * q
    axpby(n, -alpha, q, 1.0, r);

    std::swap(rho_old, rho);
  }// end of while-loop

  // clean up
  delete [] p;
  delete [] q;
  delete [] r;
  // return number of iterations and achieved residual
  *resNorm = rho;
  *numIter = iter;
  return;
}

void cg_solver_it(stencil3d const* op, int n,int nIter, double* x, double const* b,int verbose)
{
  if (op->nx * op->ny * op->nz != n)
  {
    std::cout<<n<<std::endl;
    std::cout<<op->nx<<std::endl;
    std::cout<<op->ny<<std::endl;
    std::cout<<op->nz<<std::endl;
    throw std::runtime_error("mismatch between stencil and vector dimension passed to cg_solver");
  }

  double *p = new double[n];
  double *q = new double[n];
  double *r = new double[n];

  double alpha, beta, rho=1.0, rho_old=0.0;

  // r = op * x
  apply_stencil3d_parallel(op, x, r);
  
  // r = b - r;
  axpby(n, 1.0, b, -1.0, r);

  // p = q = 0
  init(n, p, 0.0);
  init(n, q, 0.0);

  // start CG iteration
  int iter = -1;
  while (true)
  {
    iter++;

    // rho = <r, r>
    rho = dot(n,r,r);

    if (verbose)
    {
      std::cout << std::setw(4) << iter << "\t" << std::setw(8) << std::setprecision(4) << rho << std::endl;
      // std::cout << std::setprecision(4) << rho << ","; // if you want to plot it in python
    }

    // check for convergence or failure
    if ((iter > nIter)||(rho==0.0))
    {
      break;
    }

    if (rho_old==0.0)
    {
      alpha = 0.0;
    }
    else
    {
      alpha = rho / rho_old;
    }
    // p = r + alpha * p
    axpby(n, 1.0, r, alpha, p);

    // q = op * p
    apply_stencil3d_parallel(op, p, q);
    
    // beta = <p,q>
    beta = dot(n, p, q);

    alpha = rho / beta;

    // x = x + alpha * p
    axpby(n, alpha, p, 1.0, x);

    // r = r - alpha * q
    axpby(n, -alpha, q, 1.0, r);

    std::swap(rho_old, rho);
  }// end of while-loop
  
  // clean up
  delete [] p;
  delete [] q;
  delete [] r;
  return;
}