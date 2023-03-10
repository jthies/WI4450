#include "operations.hpp"
#include <omp.h>

void init(int n, double* x, double value)
{
  #pragma omp parallel for 
  for (int i=0; i<n; i++)
     x[i] = value;
  return;
}

double dot(int n, double const* x, double const* y)
{
  double res;
  #pragma omp parallel for reduction(+:res)
  for (int i=0; i<n; i++)
     res += x[i]*y[i];
  return res;
}

void axpby(int n, double a, double const* x, double b, double* y)
{
  #pragma omp parallel for
  for (int i=0; i<n; i++)
     y[i] = (a * x[i] + b * y[i]);
  return;
}

//! apply a 7-point stencil to a vector
void apply_stencil3d(stencil3d const* S,
        double const* u, double* v)
{
  int nx=S->nx, ny=S->ny, nz=S->nz;
  double ele; // value of v[S->index_c(i,j,k)]
  
  #pragma omp parallel for collapse(3) reduction(+:ele)
  for (int k=0; k<nz; k++) 
     for (int j=0; j<ny; j++)
        for(int i=0; i<nx; i++) 
	{
	   ele  = S->value_c * u[S->index_c(i,j,k)];

           if (i==0||i==nx-1) 
	      ele += (i==0) ? 
	         (u[S->index_e(i,j,k)] * S->value_e): 
		 (u[S->index_w(i,j,k)] * S->value_w);
	   else 
	      ele += u[S->index_e(i,j,k)]*S->value_e + 
			    u[S->index_w(i,j,k)]*S->value_w; 	

           if (j==0||j==ny-1) 
	      ele += (j==0) ? 
	         (u[S->index_n(i,j,k)] * S->value_n): 
		 (u[S->index_s(i,j,k)] * S->value_s);
	   else 
	      ele += u[S->index_n(i,j,k)]*S->value_n + 
			    u[S->index_s(i,j,k)]*S->value_s; 	

	  
           if (k==0||k==nz-1) 
	      ele += (k==0) ? 
	         (u[S->index_t(i,j,k)] * S->value_t): 
		 (u[S->index_b(i,j,k)] * S->value_b);  
	   else 
	      ele += u[S->index_t(i,j,k)]*S->value_t + 
		            u[S->index_b(i,j,k)]*S->value_b; 	

	   v[S->index_c(i,j,k)] = ele;
	}
 
  return;
}

