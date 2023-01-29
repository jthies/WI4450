#include "gtest_mpi.hpp"

#include "operations.hpp"

#include <iostream>

// note: you may add any number of tests to verify
// your code behaves correctly, but do not change
// the existing tests.

TEST(stencil, bounds_check)
{
  stencil3d S;
  S.nx=5;
  S.ny=3;
  S.nz=2;
  EXPECT_THROW(S.index_c(-1,0,0), std::runtime_error);
  EXPECT_THROW(S.index_c(S.nx,0,0), std::runtime_error);
  EXPECT_THROW(S.index_c(0,-1,0), std::runtime_error);
  EXPECT_THROW(S.index_c(0,S.ny,0), std::runtime_error);
  EXPECT_THROW(S.index_c(0,0,-1), std::runtime_error);
  EXPECT_THROW(S.index_c(0,0,S.nz), std::runtime_error);
}

TEST(stencil, index_order_kji)
{
  stencil3d S;
  S.nx=50;
  S.ny=33;
  S.nz=21;

  int i=10, j=15, k=9;

  EXPECT_EQ(S.index_c(i,j,k), S.index_c(i-1,j,k)+1);
  EXPECT_EQ(S.index_c(i,j,k), S.index_c(i,j-1,k)+S.nx);
  EXPECT_EQ(S.index_c(i,j,k), S.index_c(i,j,k-1)+S.nx*S.ny);
}

TEST(operations, init) {
  double res;
  const int n=15;
  double x[n];
  double val=1.0;
  init(n, x, val);

  double err=0.0;
  for (int i=0; i<n; i++) err += std::abs(x[i]-val);

  EXPECT_NEAR(res, 0.0, std::numeric_limits<double>::epsilon());
}


TEST(operations, dot) {
  const int n=150;
  double x[n], y[n];

  for (int i=0; i<n; i++)
  {
    x[i] = double(i+1);
    y[i] = 1.0/double(i+1);
  }

  double res = dot(n, x, y);
  EXPECT_NEAR(res, (double)n, n*std::numeric_limits<double>::epsilon());
}

