#include "preconditioned_gmres.hpp"
#include "operations.hpp"
#include "gmres.hpp"
#include "timer.hpp"
#include "fe.hpp"
#include "be.hpp"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdexcept>
#include <limits>

// Main program that solves the time integration in 3D
// on a unit cube. The grid size (nx,ny,nz) can be 
// passed to the executable like this:
//
// ./main_time_integration <nx> <ny> <nz>
//
// or simply ./main_time_integration <nx> for ny=nz=nx.
// If no arguments are given, the default nx=ny=nz=128 is used.
//
// Boundary conditions and forcing term f(x,y,z) are
// hard-coded in this file. See README.md for details
// on the PDE and boundary conditions.

stencil3d laplace3d_stencil(int nx, int ny, int nz)
{
  if (nx<=2 || ny<=2 || nz<=2) throw std::runtime_error("need at least two grid points in each direction to implement boundary conditions.");
  stencil3d L;
  L.nx=nx; L.ny=ny; L.nz=nz;
  double dx=1.0/(nx-1), dy=1.0/(ny-1), dz=1.0/(nz-1);
  L.value_c = -2.0/(dx*dx) - 2.0/(dy*dy) - 2.0/(dz*dz);
  L.value_n = 1.0/(dy*dy);
  L.value_e = 1.0/(dx*dx);
  L.value_s = 1.0/(dy*dy);
  L.value_w = 1.0/(dx*dx);
  L.value_t = 1.0/(dz*dz);
  L.value_b = 1.0/(dz*dz);
  return L;
}

int main(int argc, char* argv[])
{
  {Timer t("main");
  int nx = 10;
  int ny = 10;
  int nz = 10;
  // total number of unknowns
  int n=nx*ny*nz;
  double dx=1.0/(nx-1), dy=1.0/(ny-1), dz=1.0/(nz-1);
  // Laplace operator
  stencil3d L = laplace3d_stencil(nx,ny,nz);

  // solve the linear system of equations using parallel forward/backward euler
  double total_time = 10; //in seconds
  
  // Compute the 'real' solution with FE and delta_t=1e-5
  int numIter_sol=0, maxIter_sol=150;
  double resNorm_sol=1e6, epsilon_sol=1e-6; //std::sqrt(std::numeric_limits<double>::epsilon());
  double deltaT_sol = 1e-4;
  int T_sol_steps = total_time/deltaT_sol;
  int T_sol = T_sol_steps + 1;

  // initial value: initial value for the time integration method included in the rhs
  double *b_sol = new double[n*T_sol];
  init(n*T_sol, b_sol, 0.0);
  init(n, b_sol, 1.0);

  // solution vector: start with a 0 vector
  double* x_sol = new double[n*T_sol];
  init(n*T_sol, x_sol, 0.0);
  try {
  Timer t("Forward Euler solution");
  forward_euler(n,T_sol,maxIter_sol,epsilon_sol,deltaT_sol,b_sol,x_sol,&resNorm_sol,&L);
  } catch(std::exception e)
  {
    std::cerr << "Caught an exception in time_integation: " << e.what() << std::endl;
    exit(-1);
  }
  
  // Compute the approximate solution with a method of choice
  int numIter=0, maxIter=150;
  double resNorm=1e6, epsilon=1e-6; //std::sqrt(std::numeric_limits<double>::epsilon());
  double deltaT = 1e-2;
  int T_steps = total_time/deltaT;
  int T = T_steps + 1;

  // initial value: initial value for the time integration method included in the rhs
  double *b = new double[n*T];
  init(n*T, b, 0.0);
  init(n, b, 1.0);

  // solution vector: start with a 0 vector
  double* x = new double[n*T];
  init(n*T, x, 0.0);
  try {
  Timer t("Time-integration method");
  // backward_euler(n,T,maxIter,epsilon,deltaT,b,x,&resNorm,&L);
  gmres(n,T,maxIter,epsilon,deltaT,b,x,&resNorm,&L);
  // perturb_gmres(n,T,maxIter,epsilon,deltaT,b,x,&resNorm,&L);
  // jacobi_gmres(n,T,maxIter,epsilon,deltaT,b,x,&resNorm,&L);
  } catch(std::exception e)
  {
    std::cerr << "Caught an exception in time_integation: " << e.what() << std::endl;
    exit(-1);
  }

  // Calculate error
  double error = 0.0;
  for(int i=0; i<n; i++){
    error += std::abs(x_sol[n*T_sol_steps+i]*x[n*T_steps+i]);
  }
  // for (int i = 0; i<T_steps; i++){
  //   int step = (int) T_sol_steps/T_steps;
  //   for(int j = 0; j<n; j++){
  //     error += x_sol[i*step*n+j]*x[i*n+j];
  //   }
  // }
  std::cout << "error =" << std::sqrt(error) << std::endl;
  
  
  delete [] x_sol;
  delete [] x;
  }
  Timer::summarize();

  return 0;
}


//TO DO: same inputs and return the solution so we can compare later