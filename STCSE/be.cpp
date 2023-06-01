#include "fe.hpp"
#include "operations.hpp"
#include "cg_solver.hpp"
#include <iostream>
#include <cmath>
#include <omp.h>


void backward_euler(const int n,const int T,const int maxIter,const double epsilon,const double deltaT,const double* b,const double* x0,const stencil3d* L){
    
    double* x_k_min_one = new double[n];
    double* x_k = new double[n];
    double* x = new double[n*T];
    double* Ax = new double[n*T];
    double resNorm;
    int numIter;
    // Copy initial solution into the solution vector
    #pragma omp parallel for
    for (int i = 0; i<n; i++){
        x[i] = b[i];
    }
    vec2vec(n,b,x_k_min_one);
    stencil3d IdtL;
    IdtL.nx=L->nx; IdtL.ny=L->ny; IdtL.nz=L->nz;
    IdtL.value_c = 1.0 - deltaT * L->value_c;
    IdtL.value_n = -deltaT*L->value_n;
    IdtL.value_e = -deltaT*L->value_e;
    IdtL.value_s = -deltaT*L->value_s;
    IdtL.value_w = -deltaT*L->value_w;
    IdtL.value_t = -deltaT*L->value_t;
    IdtL.value_b = -deltaT*L->value_b;
    for (int i=0; i < T-1; i++){ 
        cg_solver(&IdtL, n, x_k, x_k_min_one, epsilon, 1e3, &resNorm, &numIter, 1);
        #pragma omp parallel for
        for (int j = 0; j<n; j++){
            x[(i+1)*n+j] = x_k[j];
        }
        // x_{k-1} = x_k
        vec2vec(n,x_k,x_k_min_one);
    }
    Ax_apply_stencil_backward_euler(n,T,deltaT,x,Ax,L);
    axpby(n*T,1.0,b,-1.0,Ax);
    double b_norm = sqrt(dot(n*T,b,b));
    double rel_r_norm = sqrt(dot(n*T,Ax,Ax))/b_norm;
    std::cout <<"Relative residual: "<<rel_r_norm <<std::endl;
    delete [] x_k_min_one;
    delete [] x_k;
    delete [] x;
    delete [] Ax;
}
    