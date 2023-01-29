
#include "cg_solver.hpp"
#include "operations.hpp"

#include <stdexcept>

void cg_solver(stencil3d const* op, int n, double* x, double const* b,
        double tol, int maxIter)
{
  if (op->nx * op->ny * op->nz != n)
  {
    throw std::runtime_error("mismatch between stencil and vector dimension passed to cg_solver");
  }

  double *p = new double[n];
  double *q = new double[n];
  double *r = new double[n];

  double alpha, rho=1.0, rho_old=0.0;

  // r = op * x
  // [...]
                    x_values, r);
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

    // check for convergence or failure
    if ((std::sqrt(rho) < crit.tol) || (iter > maxIter))
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

    alpha = rho / beta

    // x = x + alpha * p
    // [...]

    // r = r - alpha * q
    // [...]

    std::swap(rho_old, rho);
  }// end of while-loop

  // TODO: return iter and/or res=sqrt(rho)?


  // clean up
  delete [] p;
  delete [] q;
  delete [] r;
}
