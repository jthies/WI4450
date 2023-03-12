#include "gtest_mpi.hpp"

#include "operations.hpp"

#include "cg_solver_preconditioned.hpp"

#include <iostream>

#include <cmath>

#include <limits>


TEST(precond_cg_example, identity)
{
  const int nx=2, ny=2, nz=2;
  const int n = nx * ny * nz;
  double *x = new double[n];
  double *b = new double[n];
  double tol = std::sqrt(std::numeric_limits<double>::epsilon());
  int maxIter = n;
  double resNorm;
  int numIter;

  stencil3d S;
  S.nx = nx; S.ny = ny; S.nz = nz;
  S.value_c = 1.0;
  S.value_n = 0.0;
  S.value_e = 0.0;
  S.value_s = 0.0;
  S.value_w = 0.0;
  S.value_b = 0.0;
  S.value_t = 0.0;

  init(n, x, 1.0);
  init(n, b, 2.0);
  precond_cg_solver(&S, n, x, b, tol, maxIter, &resNorm, &numIter, 0);
  if (resNorm >= tol && numIter >= maxIter)
  {
    std::cout<<"Maximum number of iterations reached, Residual norm = " <<resNorm<<" >= " << tol << " = Tolerance: convergence not reached"<<std::endl;
  }
  else EXPECT_NEAR(0.0, resNorm, tol);
  //std::cout<<"matrix3d is a identity matrix and b is a vector with all elements equal to 2"<<std::endl;
  //std::cout<<"cg value for x, "<< "true value for x"<<std::endl;
  //for (int i=0; i<n; i++) std::cout<<x[i]<<", 2 "<<std::endl;

  delete [] x;
  delete [] b;
}

TEST(precond_cg_example, stencil)
{
  const int nx=2, ny=2, nz=2;
  const int n = nx * ny * nz;
  double *x = new double[n];
  double *b = new double[n];
  double tol = std::sqrt(std::numeric_limits<double>::epsilon());
  int maxIter = n;
  double resNorm;
  int numIter;

  stencil3d S;
  S.nx = nx; S.ny = ny; S.nz = nz;
  S.value_c = 8.0;
  S.value_n = 2.0;
  S.value_e = 4.0;
  S.value_s = 2.0;
  S.value_w = 4.0;
  S.value_b = 1.0;
  S.value_t = 1.0;

  init(n, x, 1.0);
  init(n, b, 0.0);
  precond_cg_solver(&S, n, x, b, tol, maxIter, &resNorm, &numIter, 0);
  if (resNorm >= tol && numIter >= maxIter)
  {
    std::cout<<"Maximum number of iterations reached, Residual norm = " <<resNorm<<" >= " << tol << " = Tolerance: convergence not reached"<<std::endl;
  }
  else EXPECT_NEAR(0.0, resNorm, tol);
  delete [] x;
  delete [] b;
}
