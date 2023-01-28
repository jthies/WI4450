#pragma once

void init(int n, double* x, double alue);

double dot(int n, double const* x, double const* y);

void axpby(int n, double a, double const* x, double b, double const* y, double* z);

inline int index2d(int nx, int ny, int i, int j)
{
  return j*nx + i;
}

inline int index3d(int nx, int ny, int nz, int i, int j, int k)
{
  return (k*ny +j)*nx + i;
}

//! apply the 5-point finite difference stencil
//! representing the Laplace operator on an nx x ny
//! grid. In each direction, the grid points are given
//! by
//!
//! x=0                           x=1
//!  |-----|-----| ... |-----|-----|
//! i=0   i=1   i=2   i=n-2 i=n-1 i=n
void apply_op_lapl2D(int nx, int ny,
        double const* u, double* v);

void apply_op_lapl3D(int nx, int ny, int nz,
        double const* u, double* v);

