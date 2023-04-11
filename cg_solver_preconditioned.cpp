#include "cg_solver_preconditioned.hpp"
#include "operations.hpp"

#include <cmath>
#include <stdexcept>

#include <iostream>
#include <iomanip>

//preconditioned cg solver
void precond_cg_solver(stencil3d const* op, int n, double* x, double const* b,
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
  //extra vector for preconditioning
  double *z = new double[n];

  double alpha, beta, rho=1.0, rho_old=0.0;

  // r = op * x
  apply_stencil3d(op, x, r);

  // r = b - r;
  axpby(n, 1.0, b, -1.0, r);

  init(n, p, 0.0);
  init(n, q, 0.0);

  // start CG iteration
  int iter = -1;
  while (true)
  {
    iter++;
    //solve Mz = r, with M the Jacobian preconditioner
    //z = s*r, s=1.0/op->value_c
    apply_diagonalMatrix(n, 1.0/op->value_c, r, z);

    // rho = <r, r>
    rho = dot(n,r,z);

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
    // p = z + alpha * p
    axpby(n, 1.0, z, alpha, p);

    // q = op * p
    apply_stencil3d(op, p, q);

    // beta = <p,q>
    beta = dot(n,p,q);

    alpha = rho / beta;

    // x = x + alpha * p
    axpby(n,alpha,p,1.0,x);

    // r = r - alpha * q
    axpby(n,-alpha, q, 1.0, r);

    std::swap(rho_old, rho);
  }// end of while-loop

  // clean up
  delete [] p;
  delete [] q;
  delete [] r;
  //clean up extra vector
  delete [] z;

  // return number of iterations and achieved residual
  *resNorm = rho;
  *numIter = iter;
  return;
}
