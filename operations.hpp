#pragma once

#include <sstream>
#include <stdexcept>

//////////////////////////////////
// Vector operations            //
//////////////////////////////////

// index of two dimensional array
int index(int x, int y, int no_rows);

// initialize a vector with a constant value, x[i] = value for 0<=i<n
void init(int n, double* x, double value);

// scalar product: return sum_i x[i]*y[i] for 0<=i<n
double dot(int n, double const* x, double const* y);

// vector update: compute y[i] = a*x[i] + b*y[i] for 0<=i<n
void axpby(int n, double a, double const* x, double b, double* y);

void yAx(int n, double* y, double* A, double* x);

//////////////////////////////////
// Linear operator application  //
//////////////////////////////////

// struct to represent a 3D 7-point stencil:
//
//          T  N
//          | /
//          |/
//      W---C---E
//         /|
//        / |
//       S  B
//
//   _ 
// ^ /y
//z|/  
// +-->
//   x 
//
typedef struct stencil3d
{
  // grid dimensions
  int nx, ny, nz;
  // multiplication factors when applying the stencil
  // (=sparse matrix  entries)
  double value_c, value_n, value_e, value_s, value_w, value_b, value_t;

  ////////////////////////
  // indexing functions //
  ////////////////////////

  //return the position in a vector where grid cell (i,j,k) is located
  inline int index_c(int i, int j, int k) const
  {
    if (i<0 || i>=nx || j<0 || j>=ny || k<0 || k>=nz)
    {
      std::stringstream ss;
      ss << "stencil3d index ("<<i<<","<<j<<","<<k<<") outside range ("<<nx<<","<<ny<<","<<nz<<")";
      throw std::runtime_error(ss.str());
    }
    return (k*ny +j)*nx + i;
  }

  //return the position in a vector where grid cell (i,j+1,k) is located
  inline int index_n(int i, int j, int k) const {return index_c(i,   j+1, k);};
  //return the position in a vector where grid cell (i+1,j,k) is located
  inline int index_e(int i, int j, int k) const {return index_c(i+1, j,   k);};
  //return the position in a vector where grid cell (i,j-1,k) is located
  inline int index_s(int i, int j, int k) const {return index_c(i,   j-1, k);};
  //return the position in a vector where grid cell (i-1,j,k) is located
  inline int index_w(int i, int j, int k) const {return index_c(i-1, j,   k);};
  //return the position in a vector where grid cell (i,j,k+1) is located
  inline int index_t(int i, int j, int k) const {return index_c(i,   j, k+1);};
  //return the position in a vector where grid cell (i,j,k-1) is located
  inline int index_b(int i, int j, int k) const {return index_c(i,   j, k-1);};
} stencil3d;

//! apply a 7-point stencil to a vector, v = op*u in parallel
void apply_stencil3d_parallel(stencil3d const* op,
        double const* u, double* v);

//! apply a 7-point stencil to a vector, v = op*u
void apply_stencil3d(stencil3d const* op,
        double const* u, double* v);

//! apply a 7 point stencil for T timesteps in parallel
// Calculate Ax when A consists of (-I+delta_t*op) on the lower diagonal and I on the diagonal
void Ax_apply_stencil(const stencil3d *op, const double *x, double *Ax, int T, int n, double delta_t);


