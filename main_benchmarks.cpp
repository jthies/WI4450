#include "operations.hpp"
#include "timer.hpp"
#include<iostream>



void strongScaling_fitting(double s, int N)
{
  double speedUp;
  //double s;
  double p;

  p = 1 - s;
  speedUp = 1 / (s + p / N); //Amdahl's law
}

void weakScaling_fitting(double s, int N)
{
  double speedUp;
  //double s;
  double p;

  p = 1 - s;
  speedUp = s + p * N; //Gustafson's law
}

int main(int argc, char* argv[])
{
  int nx, ny, nz;
  if (argc == 1) {nx=128; ny = 128; nz = 128;}
  else if (argc == 2) {nx=atoi(argv[1]); ny = nx; nz = nx;}
  else if (argc == 4) {nx = atoi(argv[1]); ny = atoi(argv[2]); nz = atoi(argv[3]);}
  else {std::cerr << "Invalid number of arguments (should be 0, 1 or 3)" << std::endl; exit(-1);}
  if (ny < 0) ny=nx;
  if (nz < 0) nz=nx;

  int n = nx * ny * nz;
  double x[n];
  double y[n];
  double res_value;
  double res_vector[n];
  {
    Timer t("init");
    init(n, x, 1.0);
  }
  {
    Timer t("init");
    init(n, y, 2.0);
  }
  {
    Timer t("dot");
    res_value = dot(n, x, y);
  }

  {
    Timer t("axpby");
    axpby(n, 2.5, x, 1.5, y);
  }

  stencil3d S;
  S.nx = nx; S.ny = ny; S.nz = nz;
  S.value_c = 8.0;
  S.value_e = 4.0;
  S.value_s = 2.0;
  S.value_w = 4.0;
  S.value_n = 2.0;
  S.value_t = 1.0;
  S.value_b = 1.0;

  {
    Timer t("apply_stencil3d");
    apply_stencil3d(&S, x, res_vector);
  }

  Timer::summarize();

  return 0;
}
