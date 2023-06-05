#include "operations.hpp"
#include "gmres.hpp"
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <omp.h>


void arnoldi(const int n, const int T, const int maxIter,const int j,const double deltaT, double* Q, double* H,const stencil3d* L){
    double* Q_j = new double[n*T]();
    double* AQ_j = new double[n*T]();
    //Q_j = Q[:,j] 
    matrix2vec(n*T,j,Q_j,Q);
    //Q[:,j+1] = AQ_j
    Ax_apply_stencil_backward_euler(n, T, deltaT, Q_j, AQ_j, L);
    //Ax_apply_stencil_forward_euler(n, T, deltaT, Q_j, AQ_j, L);
    vec2matrix(n*T,j+1,AQ_j,Q);
    for (int i = 0; i < j + 1; i++) {
        // H[i][j] = Q[:][i]^T*Q[:,j+1]
        H[index(i, j, maxIter+1)] = matrix_col_vec_dot(n*T, i, AQ_j, Q);
        // Q[:][j+1] = Q[:][j+1] - H[i][j]*Q[:,i]
        orthogonalize_Q(n*T,maxIter+1,i,j+1,Q,H);
    }
    H[index(j + 1, j, maxIter+1)] = sqrt(matrix_col_dot(n*T,j+1,Q));
    matrix_col_scale(n*T,j+1,H[index(j + 1, j, maxIter+1)],Q);
    delete[] Q_j;
    delete[] AQ_j;
}

void givens(const int j, const int maxIter, double* cos, double* sin, double* H){
    double H_temp,denom;
    for(int i=0; i<j; i++){
        H_temp = cos[i] * H[index(i,j,maxIter+1)] + sin[i] * H[index(i+1,j,maxIter+1)];
        H[index(i+1,j,maxIter+1)] = -sin[i] * H[index(i,j,maxIter+1)]+ cos[i] * H[index(i+1,j,maxIter+1)];
        H[index(i,j,maxIter+1)] = H_temp;
    }
    denom = sqrt(H[index(j,j,maxIter+1)] *H[index(j,j,maxIter+1)] + H[index(j+1,j,maxIter+1)]*H[index(j+1,j,maxIter+1)]);
    cos[j]= H[index(j,j,maxIter+1)]/denom;
    sin[j]= H[index(j+1,j,maxIter+1)]/denom;
    H[index(j,j,maxIter+1)] = cos[j]*H[index(j,j,maxIter+1)]+sin[j] * H[index(j+1,j,maxIter+1)];
    H[index(j+1,j,maxIter+1)] = 0.0;
}

void backward_substitution(const int iter, const int maxIter, const double* e_1,const double* H, double* y){
    y[iter] = e_1[iter]/H[index(iter, iter, maxIter+1)];
    for (int i=(iter-1); i>=0; i--){
        y[i] = e_1[i];        
        for (int j=i+1; j <= iter; j++){
            y[i] -= H[index(i, j, maxIter+1)]*y[j];
        }           
        y[i] = y[i] / H[index(i, i, maxIter+1)];
    }
}

void gmres(const int n,const int T,const int maxIter,const double epsilon, const double deltaT,const double* b, const double* x0, double* resNorm, const stencil3d* L){
    std::cout<<"Starting GMRES..."<<std::endl;
    double* r_0 = new double[n*T]();
    double Q[n * T * (maxIter+1)] = {0.0};
    double H[(maxIter+1)*maxIter] = {0.0};
    double cos[maxIter] = {0.0};
    double sin[maxIter] = {0.0};
    double e_1[maxIter+1] = {0.0};
    double y[n*T] = {0.0};
    double x[n*T] = {0.0};
    double Ax[n*T] = {0.0};
    int iter;

    //b_norm = ||b||_2
    double b_norm = sqrt(dot(n*T,b,b));
    //r_0 = b - Ax_0
    Ax_apply_stencil_backward_euler(n, T, deltaT, x0, r_0, L);
    //Ax_apply_stencil_forward_euler(n, T, deltaT, x0, r_0, L);
    axpby(n*T,1.0,b,-1.0,r_0);
    //Q[:,0] = r_0/||r_0||_2
    double r0_norm = sqrt(dot(n*T,r_0,r_0));
    vec2matrix(n*T,0,r_0,Q);
    matrix_col_scale(n*T, 0, r0_norm, Q);
    //e[0] = r0_norm
    e_1[0] = r0_norm;
    //arnoldi iteration
    for (int j = 0; j<maxIter; j++){
        arnoldi(n,T, maxIter, j,deltaT,Q,H,L);
        givens(j,maxIter, cos, sin, H);
        e_1[j+1] = -sin[j]*e_1[j];
        e_1[j] = cos[j]*e_1[j];
        std::cout << "Iter: " << j << " Error: " << std::abs(e_1[j+1])/b_norm << std::endl;
        if ((std::abs(e_1[j+1])/b_norm < epsilon) || (j==maxIter-1)){
            iter=j;
            std::cout <<"GMRES Stopped"<< std::endl;
            std::cout <<"Iteration:         "<<j << std::endl;
            std::cout <<"Epsilon:           "<<epsilon << std::endl;
            std::cout <<"Relative residual: "<<std::abs(e_1[j+1])/b_norm <<" (std::abs(e_1[j+1])/b_norm)"<< std::endl;
            break;
        }
    }
    // Backward substitution
    backward_substitution(iter, maxIter, e_1,H,y);
    // x^{*} = Qy + x0 
    matrix_vec_prod(n*T, iter+2, x, Q, y);
    axpby(n * T, 1.0, x0, 1.0, x);
    // r = b-Ax
    Ax_apply_stencil_backward_euler(n, T, deltaT, x, Ax, L);
    //Ax_apply_stencil_forward_euler(n, T, deltaT, x, Ax, L);
    axpby(n*T, 1.0, b, -1.0, Ax);
    // rel_r_norm = ||r||_2/||b||_2 
    double rel_r_norm = sqrt(dot(n*T, Ax, Ax))/b_norm;
    std::cout <<"Relative residual: "<<rel_r_norm <<" (||r||_2/||r_0||_2)"<< std::endl;
    *resNorm = rel_r_norm;
    delete [] r_0;
    std::cout<<"GMRES Finished"<<std::endl;
}






