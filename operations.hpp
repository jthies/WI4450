#pragma once

#include <sstream>
#include <stdexcept>

//////////////////////////////////
// Vector operations            //
//////////////////////////////////

void init(int n, double* x, double alue);

double dot(int n, double const* x, double const* y);

void axpby(int n, double a, double const* x, double b, double const* y, double* z);

//////////////////////////////////
// Linear operator application  //
//////////////////////////////////

typedef struct stencil3d
{
  int nx, ny, nz;
  double center, north, east, south, west, bottom, top;

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

  inline int index_n(int i, int j, int k) const {return index_c(i,   j+1, k);};
  inline int index_e(int i, int j, int k) const {return index_c(i+1, j,   k);};
  inline int index_s(int i, int j, int k) const {return index_c(i,   j-1, k);};
  inline int index_w(int i, int j, int k) const {return index_c(i-1, j,   k);};
  inline int index_b(int i, int j, int k) const {return index_c(i,   j, k+1);};
  inline int index_t(int i, int j, int k) const {return index_c(i,   j, k-1);};

} stencil3d;


//! apply a 7-point stencil to a vector
void apply_stencil3d(stencil3d const* S,
        double const* u, double* v);

