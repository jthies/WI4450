#pragma once

//////////////////////////////////
// Vector operations            //
//////////////////////////////////

void init(int n, double* x, double alue);

double dot(int n, double const* x, double const* y);

void axpby(int n, double a, double const* x, double b, double const* y, double* z);

//////////////////////////////////
// Linear operator application  //
//////////////////////////////////

typedef struct
{
  int nx, ny, nz;
  double center, north, east, south, west, bottom, top;
} stencil3d;

inline int index3d(int nx, int ny, int nz, int i, int j, int k)
{
  return (k*ny +j)*nx + i;
}

//! apply a 7-point stencil to a vector
void apply_stencil3d(stencil const* S,
        double const* u, double* v);

