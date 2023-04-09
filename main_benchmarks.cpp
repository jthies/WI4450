#include "operations.hpp"
#include "timer.hpp"
#include<iostream>

int main(int argc, char* argv[])
{
  int nx, ny, nz;
  int iter;
  if (argc == 1) {nx=128; ny = 128; nz = 128; iter = 128;}
  else if (argc == 3) {nx=atoi(argv[1]); ny = nx; nz = nx; iter = atoi(argv[2]);}
  else if (argc == 5) {nx = atoi(argv[1]); ny = atoi(argv[2]); nz = atoi(argv[3]); iter = atoi(argv[4]);}
  else {std::cerr << "Invalid number of arguments (should be 0, 1 or 3)" << std::endl; exit(-1);}
  if (ny < 0) ny=nx;
  if (nz < 0) nz=nx;

  int n = nx * ny * nz;
  double x[n];
  double y[n];
  double res_value;
  double res_vector[n];
  stencil3d S;
  S.nx = nx; S.ny = ny; S.nz = nz;
  S.value_c = 8.0;
  S.value_e = 4.0;
  S.value_s = 2.0;
  S.value_w = 4.0;
  S.value_n = 2.0;
  S.value_t = 1.0;
  S.value_b = 1.0;
  int flp;
  for (int i=0; i<iter; i++)
  {
      {
        Timer t("init");
        t.m = 0.0; 
        t.b = 1.0 * sizeof(double) * n;
        init(n, x, 1.0);
      }
      {
        Timer t("init");
        t.m = 0.0; 
        t.b = 1.0 * sizeof(double) * n;
        init(n, y, 2.0);
      }
      {
        Timer t("dot");
        t.m = 2.0 * n;
        t.b = 2.0 * sizeof(double) * n;
        res_value = dot(n, x, y);
      }
    
      {
        Timer t("axpby");
        t.m = 3.0 * n;
        t.b = 3.0 * sizeof(double) * n ;
        axpby(n, 2.5, x, 1.5, y);
      }
    
      {
        Timer t("apply_stencil3d");
        t.m = 2.0 * n;                                                                                                                                        //possibly large value (encoutered at n=600^3)
        t.b = 1.0 * sizeof(S) + 2.0 * n * sizeof(double);
        apply_stencil3d(&S, x, res_vector);
      }
  }

  Timer::summarize();

  return 0;
}
