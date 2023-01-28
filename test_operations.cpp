#include "gtest_mpi.hpp"

#include "operations.hpp"

#include <iostream>

TEST(operations, init) {
  double res;
  const int n=15;
  double x[n];
  double val=1.0;
  init(n, x, val);

  double err=0.0;
  for (int i=0; i<n; i++) err += std::abs(x[i]-val);

  ASSERT_NEAR(res, 0.0, std::numeric_limits<double>::epsilon());
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
  ASSERT_NEAR(res, (double)n, n*std::numeric_limits<double>::epsilon());
}

