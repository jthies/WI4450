#include "fe.hpp"
#include "operations.hpp"
#include <iostream>
#include <cmath>
#include <omp.h>

void forward_euler(const int n,const int T,const int maxIter,const double epsilon,const double deltaT,const double* b, double* x, double* resNorm,const stencil3d* L){
    
    double* x_k_min_one = new double[n];
    double* x_k = new double[n];
    double* Ax = new double[n*T];
    // Copy initial solution into the solution vector
    #pragma omp parallel for
    for (int i = 0; i<n; i++){
        x[i] = b[i];
    }
    vec2vec(n,b,x_k_min_one);
    for (int i=0; i < T-1; i++){
        // x_k=L*x_{k-1}
        apply_stencil3d_parallel(L,x_k_min_one,x_k);
        // x_k = x_{k-1} + deltaT*x_k
        axpby(n,1.0,x_k_min_one,deltaT,x_k);
        //x[(i+1)*n:(i+2)*n] = x_k
        #pragma omp parallel for
        for (int j = 0; j<n; j++){
            x[(i+1)*n+j] = x_k[j];
        }
        // x_{k-1} = x_k
        vec2vec(n,x_k,x_k_min_one);
    }
    Ax_apply_stencil_forward_euler(n,T,deltaT,x,Ax,L);
    axpby(n*T,1.0,b,-1.0,Ax);
    double b_norm = sqrt(dot(n*T,b,b));
    double rel_r_norm = sqrt(dot(n*T,Ax,Ax))/b_norm;
    std::cout <<"Relative residual: "<<rel_r_norm <<std::endl;
    *resNorm = rel_r_norm;
    delete [] x_k_min_one;
    delete [] x_k;
    delete [] x;
    delete [] Ax;
}
