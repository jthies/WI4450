#include "cg_solver_preconditioned.hpp"
#include "operations.hpp"

#include <cmath>
#include <stdexcept>

#include <iostream>
#include <iomanip>

//preconditioned
void precond_cg_solver(stencil3d const* op, int n, double* x, double const* b,
        double tol, int maxIter,
        double* resNorm, int* numIter,
        int verbose)
{
  if (op->nx * op->ny * op->nz != n)
  {
    throw std::runtime_error("mismatch between stencil and vector dimension passed to cg_solver");
  }
  //double trace = n * op->value_c;
  stencil3d invM;
  invM.nx = op->nx; invM.ny = op->ny; invM.nz = op->nz;
  invM.value_c = 1.0/(op->value_c);//(op->value_c)/trace;//(op->value_c)/trace;
  invM.value_n = 0.0;
  invM.value_e = 0.0;
  invM.value_s = 0.0;//-(op->value_s)/(op->value_c)*(op->value_s)/trace; //-(op->value_s)/trace;
  invM.value_w = 0.0;//-(op->value_w)/(op->value_c)*(op->value_w)/trace; //-(op->value_w)/trace;
  invM.value_t = 0.0;
  invM.value_b = 0.0;//-(op->value_b)/(op->value_c)*(op->value_b)/trace; //-(op->value_b)/trace;

  double *p = new double[n];
  double *q = new double[n];
  double *r = new double[n];
  double *z = new double[n];

  double alpha, beta, rho=1.0, rho_old=0.0;

  // r = op * x
  // [...]
  apply_stencil3d(op, x, r);

  // r = b - r;
  // [...]
  axpby(n, 1.0, b, -1.0, r);

  init(n, p, 0.0);
  //init(n, q, 0.0);

  // start CG iteration
  int iter = -1;
  while (true)
  {
    iter++;
    //solve Mz = r, Gauss-Seidel preconditioner
    //z = invM r
    apply_stencil3d(&invM, r, z);

    // rho = <r, r>
    // [...]
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
    // [...]
    axpby(n, 1.0, z, alpha, p);

    // q = op * p
    // [...]
    apply_stencil3d(op, p, q);

    // beta = <p,q>
    // [...]
    beta = dot(n,p,q);

    alpha = rho / beta;

    // x = x + alpha * p
    // [...]
    axpby(n,alpha,p,1.0,x);

    // r = r - alpha * q
    // [...]
    axpby(n,-alpha, q, 1.0, r);

    std::swap(rho_old, rho);
  }// end of while-loop

  // clean up
  delete [] p;
  delete [] q;
  delete [] r;
  delete [] z;

  // return number of iterations and achieved residual
  *resNorm = rho;
  *numIter = iter;
  return;
}
