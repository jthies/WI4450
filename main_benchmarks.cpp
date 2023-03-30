#include "operations.hpp"
#include "timer.hpp"
#include<iostream>

double divide_over_two_units(int op)
{
    int units = 2;
    int d_op = op / units + op%units;
    return double(d_op)
}

int main(int argc, char* argv[])
{
  int nx, ny, nz;
  int iter;
  if (argc == 1) {iter = 128; nx=128; ny = 128; nz = 128;}
  else if (argc == 2) {nx=atoi(argv[1]); ny = nx; nz = nx;}
  else if (argc == 3) {nx=atoi(argv[1]); ny = nx; nz = nx; iter = atoi(argv[2]);}
  else if (argc == 4) {nx = atoi(argv[1]); ny = atoi(argv[2]); nz = atoi(argv[3]);}
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

  for (int i=0; i<iter; i++)
  {
    {
      Timer t("init");
      t.m = 0.0; 
      t.b = divide_over_two_units(1)*n*sizeof(double); //type double
      init(n, x, 1.0);
    }
    {
      Timer t("init");
      t.m = 0.0; 
      t.b = divide_over_two_units(1)*n*sizeof(double); //type double
      init(n, y, 2.0);
    }
    {
      Timer t("dot");
      t.m = divide_over_two_units(2)*n; 
      t.b = divide_over_two_units(2)*n*sizeof(double); //type double
      res_value = dot(n, x, y);
    }

    {
      Timer t("axpby");
      t.m = divide_over_two_units(3)*n; 
      t.b = divide_over_two_units(3)*n*sizeof(double); //type double
      axpby(n, 2.5, x, 1.5, y);
    }

    {
      Timer t("apply_stencil3d");
      t.m = divide_over_two_units(1) * n + divide_over_two_units(4) * ( (nx - 1.0) * ny * nz + nx * (ny - 1.0) * nz + nx * ny * (nz - 1.0) ); //type double, already within the equation, 
                                                                                                    //otherwise incorrect result because of 
                                                                                                //possibly large value (encoutered at n=600^3)
      t.b = divide_over_two_units(1)*sizeof(S) + divide_over_two_units(2)*n*sizeof(double)); //(2.0 * n + 7.0) * sizeof(double) + 3.0 * sizeof(int); //2*n+y type double, 3 type integer
      //std::cout<<sizeof(S)<<std::endl;
      apply_stencil3d(&S, x, res_vector);
    }
  }
  Timer::summarize();

  return 0;
}
