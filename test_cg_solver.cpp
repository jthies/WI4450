#include "gtest_mpi.hpp"
#include "operations.hpp"
#include "cg_solver.hpp"

#include <iostream>
#include <cmath>
#include <limits>

TEST(cg_solver, cg_solver)
{
/*This is a test for cg_solver which solves the linear system Ax=b
  Here we use a diagonal matrix as A and all-one vector as b, by which 
  we know the correct solution easily.*/
  const int nx=9, ny=5, nz=6;
  const int n=nx*ny*nz;
  
  stencil3d S;
  S.nx=nx; S.ny=ny; S.nz=nz;
  S.value_c = 2;
  S.value_n = 0;
  S.value_e = 0;
  S.value_s = 0;
  S.value_w = 0;
  S.value_b = 0;
  S.value_t = 0;

  double *x = new double[n]; // solution vector x
  double *b = new double[n]; // right hand side vector b
  init(n, x, 0.0); // solution starts with [0,0,...]
  init(n, b, 1.0); // right hand side b=[1,1,...] 

  // solve the linear system of equations using CG
  int numIter, maxIter=100;
  double resNorm, tol=std::sqrt(std::numeric_limits<double>::epsilon());

  cg_solver(&S, n, x, b, tol, maxIter, &resNorm, &numIter, 0);

  double err=0.0, solution=0.5;
  for (int i=0; i<n; i++) err = std::max(err, std::abs(x[i]-solution));
  EXPECT_NEAR(1.0+err, 1.0, std::numeric_limits<double>::epsilon());
  
  delete [] x;
  delete [] b;

}

