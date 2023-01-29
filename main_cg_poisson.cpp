#include "operations.hpp"
#include "cg_solver.hpp"

#include <cmath>

// Main program that solves the 3D Poisson equation
// on a unit cube. The grid size (nx,ny,nz) can be 
// passed to the executable like this:
//
// cg_poisson <nx> <ny> <nz>
//
// Boundary conditions and forcing term f(x,y,z) are
// hard-coded in this file. See README.md for details
// on the PDE and boundary conditions.

// Forcing term
double f(double x, double y, double z)
{
  return z*sin(2*M_PI*x)*std::sin(M_PI*y) + 8*z*z*z;
}

// boundary condition at z=0
double g_0(double x, double y)
{
  return x*(1.0-x)*y*(1-y);
}

int main()
{
  return 0;
}
