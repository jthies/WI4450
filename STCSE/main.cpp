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

  // solve the linear system of equations using parallel forward euler
  double total_time = 10.0;
  int numIter=0, maxIter=150;
  double resNorm=10e6, epsilon=1e-6; //std::sqrt(std::numeric_limits<double>::epsilon());

  Timer baseline_t("baseline");
  double deltaT_baseline = 1e-5;
  int T_baseline = total_time/deltaT_baseline;
  double *b_baseline = new double[n*T_baseline];
  init(n*T_baseline, b_baseline, 0.0);
  init(n, b_baseline, 1.0);
  double* x_baseline = new double[n*T_baseline];
  init(n*T_baseline, x_baseline, 0.0);
  forward_euler(n,T_baseline,maxIter,epsilon,deltaT_baseline,b_baseline,x_baseline,&resNorm,&L);
  std::cout<<"Norm solution: "<< dot(n*T_baseline,x_baseline,x_baseline)<<std::endl;
  delete [] b_baseline;
  baseline_t.~Timer();

  Timer fe_t("forward euler");
  double deltaT_fe = 1e-4;
  int T_fe = total_time/deltaT_fe;
  double *b_fe = new double[n*T_fe];
  init(n*T_fe, b_fe, 0.0);
  init(n, b_fe, 1.0);
  double* x_fe = new double[n*T_fe];
  init(n*T_fe, x_fe, 0.0);
  forward_euler(n,T_fe,maxIter,epsilon,deltaT_fe,b_fe,x_fe,&resNorm,&L);
  std::cout<<"Norm solution: "<< dot(n*T_fe,x_fe,x_fe)<<std::endl;
  std::cout<<"Precision FE: "<< precision(n, T_baseline, T_fe, x_baseline, x_fe)<<std::endl;
  delete [] b_fe;
  fe_t.~Timer();

  Timer be_t("backward euler");
  double deltaT_be = 1e-2;
  int T_be = total_time/deltaT_be;
  double *b_be = new double[n*T_be];
  init(n*T_be, b_be, 0.0);
  init(n, b_be, 1.0);
  double* x_be = new double[n*T_be];
  init(n*T_be, x_be, 0.0);
  backward_euler(n,T_be,maxIter,epsilon,deltaT_be,b_be,x_be,&resNorm,&L);
  std::cout<<"Norm solution: "<< dot(n*T_be,x_be,x_be)<<std::endl;
  std::cout<<"Precision BE: "<< precision(n, T_baseline, T_be, x_baseline, x_be)<<std::endl;
  delete [] b_be;
  be_t.~Timer();

  Timer gmres_t("gmres");
  double deltaT_gmres = 1e-2;
  int T_gmres = total_time/deltaT_gmres;
  double *b_gmres = new double[n*T_gmres];
  init(n*T_gmres, b_gmres, 0.0);
  init(n, b_gmres, 1.0);
  double* x_gmres = new double[n*T_gmres];
  init(n*T_gmres, x_gmres, 0.0);
  gmres(n,T_gmres,maxIter,epsilon,deltaT_gmres,b_gmres,x_gmres,&resNorm,&L);
  std::cout<<"Norm solution: "<< dot(n*T_gmres,x_gmres,x_gmres)<<std::endl;
  std::cout<<"Precision GMRES: "<< precision(n, T_baseline, T_gmres, x_baseline, x_gmres)<<std::endl;
  delete [] b_gmres;
  gmres_t.~Timer();

  Timer perturb_gmres_t("perturb_gmres");
  double deltaT_perturb_gmres = 1e-2;
  int T_perturb_gmres = total_time/deltaT_perturb_gmres;
  double *b_perturb_gmres = new double[n*T_perturb_gmres];
  init(n*T_perturb_gmres, b_perturb_gmres, 0.0);
  init(n, b_perturb_gmres, 1.0);
  double* x_perturb_gmres = new double[n*T_perturb_gmres];
  init(n*T_perturb_gmres, x_perturb_gmres, 0.0);
  perturb_gmres(n,T_perturb_gmres,maxIter,epsilon,deltaT_perturb_gmres,b_perturb_gmres,x_perturb_gmres,&resNorm,&L);
  std::cout<<"Norm solution: "<< dot(n*T_perturb_gmres,x_perturb_gmres,x_perturb_gmres)<<std::endl;
  std::cout<<"Precision Perturb GMRES: "<< precision(n, T_baseline, T_perturb_gmres, x_baseline, x_perturb_gmres)<<std::endl;
  delete [] b_perturb_gmres;
  perturb_gmres_t.~Timer();

  Timer jacobi_gmres_t("jacobi_gmres");
  double deltaT_jacobi_gmres = 1e-2;
  int T_jacobi_gmres = total_time/deltaT_jacobi_gmres;
  double *b_jacobi_gmres = new double[n*T_jacobi_gmres];
  init(n*T_jacobi_gmres, b_jacobi_gmres, 0.0);
  init(n, b_jacobi_gmres, 1.0);
  double* x_jacobi_gmres = new double[n*T_jacobi_gmres];
  init(n*T_jacobi_gmres, x_jacobi_gmres, 0.0);
  jacobi_gmres(n,T_jacobi_gmres,maxIter,epsilon,deltaT_jacobi_gmres,b_jacobi_gmres,x_jacobi_gmres,&resNorm,&L);
  std::cout<<"Norm solution: "<< dot(n*T_jacobi_gmres,x_jacobi_gmres,x_jacobi_gmres)<<std::endl;
  std::cout<<"Precision Jacobi GMRES: "<< precision(n, T_baseline, T_jacobi_gmres, x_baseline, x_jacobi_gmres)<<std::endl;
  delete [] b_jacobi_gmres;
  jacobi_gmres_t.~Timer();
  }
  Timer::summarize();
   
  return 0;
}