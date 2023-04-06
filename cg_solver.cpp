#include "timer.hpp"
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
    throw std::runtime_error("mismatch between stencil and vector dimension passed to cg_solver");
  }

  double *p = new double[n];
  double *q = new double[n];
  double *r = new double[n];

  double alpha, beta, rho=1.0, rho_old=0.0;

  // r = op * x
  { Timer t("apply_stencil3d",op->nx, op->ny, op->nz);
    apply_stencil3d(op, x, r);}

  // r = b - r;
  { Timer t("axpby",op->nx, op->ny, op->nz);
    axpby(n, 1.0, b, -1.0, r);}

  // p = q = 0
  { Timer t("init",op->nx, op->ny, op->nz);
    init(n, p, 1.0);}
  { Timer t("init",op->nx, op->ny, op->nz);
    init(n, q, 1.0);}

  // start CG iteration
  int iter = -1;
  while (true)
  {
    iter++;

    // rho = <r, r>
    { Timer t("dot",op->nx, op->ny, op->nz);
      rho = dot(n, r, r);}

    if (verbose)
    {
      std::cout << std::setw(4) << iter << "\t" << std::setw(8) << std::setprecision(4) << rho << std::endl;
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
    { Timer t("axpby",op->nx, op->ny, op->nz);
      axpby(n, 1.0, r, alpha, p);}
    
    // q = op * p
    { Timer t("apply_stencil3d",op->nx, op->ny, op->nz);
      apply_stencil3d(op, p, q);}

    // beta = <p,q>
    { Timer t("dot",op->nx, op->ny, op->nz);
      beta = dot(n, p, q);}

    alpha = rho / beta;

    // x = x + alpha * p
    { Timer t("axpby",op->nx, op->ny, op->nz);
    axpby(n, alpha, p, 1.0, x);}

    // r = r - alpha * q
    { Timer t("axpby",op->nx, op->ny, op->nz);
    axpby(n,-alpha, q, 1.0, r);}

    std::swap(rho_old, rho);
  }// end of while-loop

  // clean up
  delete [] p;
  delete [] q;
  delete [] r;

  // return number of iterations and achieved residual
  *resNorm = rho;
  *numIter = iter;

 // Timer::summarize();

  return;
}
