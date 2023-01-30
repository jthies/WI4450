
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
  // [...]

  // r = b - r;
  // [...]

  // p = q = 0
  // [...]

  // start CG iteration
  int iter = -1;
  while (true)
  {
    iter++;

    // rho = <r, r>
    // [...]

    if (verbose)
    {
      std::cout << std::setw(4) << iter << "\t" << std::setw(8) << std::setprecision(4) << rho << std::endl;
    }

    // check for convergence or failure
    if ((std::sqrt(rho) < tol) || (iter > maxIter))
    {
      break;
    }

    alpha = rho / rho_old;

    // p = r + alpha * p
    // [...]

    // q = op * p
    // [...]

    // beta = <p,q>
    // [...]

    alpha = rho / beta;

    // x = x + alpha * p
    // [...]

    // r = r - alpha * q
    // [...]

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
