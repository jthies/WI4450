#include "operations.hpp"
#include "timer.hpp"

#include <iostream>


int main(int argc, char* argv[])
{
 {Timer t("main"); // measure the total running time of main function
  int nx, ny, nz; 

  if      (argc==1) {nx=128;           ny=128;           nz=128;}
  else if (argc==2) {nx=atoi(argv[1]); ny=nx;            nz=nx;}
  else if (argc==4) {nx=atoi(argv[1]); ny=atoi(argv[2]); nz=atoi(argv[3]);}
  else {std::cerr << "Invalid number of arguments (should be 0, 1 or 3)"<<std::endl; exit(-1);}
  if (ny<0) ny=nx;
  if (nz<0) nz=nx;

  int n = nx*ny*nz; // problem size

  double* x = new double[n];   
  double* y = new double[n];

  stencil3d S;
  S.nx=nx; S.ny=ny; S.nz=nz;
  S.value_c = 6;
  S.value_n = -1;
  S.value_e = -1;
  S.value_s = -1;
  S.value_w = -1;
  S.value_b = -1;
  S.value_t = -1;

  // measure the running time of init
  { 
    Timer t("init");
    init(n,x,2.0); 
  }

  {
    Timer t("init");
    init(n,y,1.0);
  }
  
  // measure the running time of dot
  {
    Timer t("dot");
    dot(n,x,y);
  }

  // measure the running time of axpby
  { Timer t("axpby");
    axpby(n, 3.0, x, 2.0, y);
  }

  // measure the running time of apply_stencil3d
  {
    Timer t("applystencil");
    apply_stencil3d(&S, x, y);
  }

  delete [] x;
  delete [] y;}
  
  Timer::summarize();

  return 0;
 
}

