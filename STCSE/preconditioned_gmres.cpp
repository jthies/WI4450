#include "preconditioned_gmres.hpp"
#include "cg_solver.hpp"
#include "operations.hpp"
#include "gmres.hpp"

#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <omp.h>

void perturb_arnoldi(const int n, const int T, const int maxIter,const int j,const double deltaT, const double* perturb, double* Q, double* H,const stencil3d* L){
    double* Q_j = new double[n*T]();
    double* AQ_j = new double[n*T]();
    //Q_j = Q[:,j] 
    matrix2vec(n*T,j,Q_j,Q);
    //Q[:,j+1] = AQ_j
    //Ax_apply_stencil_forward_euler(n, T, deltaT, Q_j, AQ_j, L);
    Ax_apply_stencil_backward_euler(n, T, deltaT, Q_j, AQ_j, L);
    vector_scale(n*T,perturb,AQ_j);
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


void perturb_gmres(const int n,const int T,const int maxIter,const double epsilon, const double deltaT,const double* b, double* x0, double* resNorm, const stencil3d* L){
    double* r_0 = new double[n*T]();
    double Q[n * T * (maxIter+1)] = {0.0};
    double H[(maxIter+1)*maxIter] = {0.0};
    double cos[maxIter] = {0.0};
    double sin[maxIter] = {0.0};
    double e_1[maxIter+1] = {0.0};
    double y[n*T] = {0.0};
    double x[n*T] = {0.0};
    double Ax[n*T] = {0.0};
    double* perturb = new double[n*T]();
    int iter;
    srand(1);
    for (int i = 0; i<n*T;i++){
        perturb[i] = 1+ (double) rand()/RAND_MAX;
    }
    //b_norm = ||b||_2
    double b_norm = sqrt(dot(n*T,b,b));
    //r_0 = b - Ax_0
    //Ax_apply_stencil_forward_euler(n, T, deltaT, x0, r_0, L);
    Ax_apply_stencil_backward_euler(n, T, deltaT, x0, r_0, L);
    axpby(n*T,1.0,b,-1.0,r_0);
    vector_scale(n*T, perturb, r_0);
    //Q[:,0] = r_0/||r_0||_2
    double r0_norm = sqrt(dot(n*T,r_0,r_0));
    vec2matrix(n*T,0,r_0,Q);
    matrix_col_scale(n*T, 0, r0_norm, Q);
    //e[0] = r0_norm
    e_1[0] = r0_norm;
    //arnoldi iteration
    for (int j = 0; j<maxIter; j++){
        perturb_arnoldi(n,T, maxIter, j,deltaT,perturb,Q,H,L);
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
    backward_substitution(iter, maxIter, e_1,H, y);
    // x^{*} = Qy + x0 
    matrix_vec_prod(n*T, iter+2, x, Q, y);
    //vector_scale(n*T,perturb,x);
    axpby(n * T, 1.0, x0, 1.0, x);

    // r = b-Ax
    //Ax_apply_stencil_forward_euler(n, T, deltaT, x, Ax, L);
    Ax_apply_stencil_backward_euler(n, T, deltaT, x, Ax, L);
    //vector_scale(n*T,perturb,Ax);
    axpby(n*T, 1.0, b, -1.0, Ax);
    // rel_r_norm = ||r||_2/||b||_2 
    double rel_r_norm = sqrt(dot(n*T, Ax, Ax))/b_norm;
    std::cout <<"Relative residual: "<<rel_r_norm <<" (||r||_2/||r_0||_2)"<< std::endl;
    *resNorm = rel_r_norm;

    delete [] r_0;
    delete [] perturb;
}

void block_cg_solver(const int n, const int T,const double deltaT,double* x,const double* b,const stencil3d* L){
    double* cg_resNorm;
    double epsilon =std::sqrt(std::numeric_limits<double>::epsilon());
    const int n_cg_iter = 3;
    int* cg_iter;
    stencil3d D_be;
    D_be.nx=L->nx; D_be.ny=L->ny; D_be.nz=L->nz;
    D_be.value_c = 1.0 - deltaT * L->value_c;
    D_be.value_n = -deltaT* L->value_n;
    D_be.value_e = -deltaT* L->value_e;
    D_be.value_s = -deltaT* L->value_s;
    D_be.value_w = -deltaT* L->value_w;
    D_be.value_t = -deltaT* L->value_t;
    D_be.value_b = -deltaT* L->value_b;
    for (int i =0; i<T;i++){
        double temp_x[n] = {0.0};
        double* temp_b = new double[n*T]();
        for (int j = 0; j<n;j++){
            temp_b[j] = b[(i*n)+j];
        }
        cg_solver_it(&D_be,n,n_cg_iter,temp_x,temp_b,0);
        for (int j = 0; j<n;j++){
            x[(i*n)+j] = temp_x[j];
        } 
        delete [] temp_b;
    }
}


void precond_cg_arnoldi(const int n, const int T,const int maxIter,const int j,const double deltaT, const double epsilon, double* Q, double* H,const stencil3d* L){
    double* Q_j = new double[n*T]();
    double* AQ_j = new double[n*T]();
    double Q_j_p1[n*T] = {0.0};
    double cg_x0[n*T] = {0.0};
    double* cg_resNorm;
    int* cg_n_iter_;

    //Q_j = Q[:,j] 
    matrix2vec(n*T,j,Q_j,Q);
    //Q[:,j+1] = AQ_j
    //Ax_apply_stencil_forward_euler(n, T, deltaT, Q_j, AQ_j, L);
    Ax_apply_stencil_backward_euler(n, T, deltaT, Q_j, AQ_j, L);
    block_cg_solver(n, T,deltaT,Q_j_p1,AQ_j,L);
    vec2matrix(n*T,j+1,Q_j_p1,Q);
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


void jacobi_gmres(const int n,const int T,const int maxIter,const double epsilon, const double deltaT,const double* b, double* x0, double* resNorm, const stencil3d* L){
    double* r_0 = new double[n*T]();
    double Q[n * T * (maxIter+1)] = {0.0};
    double H[(maxIter+1)*maxIter] = {0.0};
    double cos[maxIter] = {0.0};
    double sin[maxIter] = {0.0};
    double e_1[maxIter+1] = {0.0};
    double y[n*T] = {0.0};
    double x[n*T] = {0.0};
    double Ax[n*T] = {0.0};
    double* resNorm_cg;
    int* n_iter_cg_;
    int iter;
    int n_cg_iter = 3;
    
    //b_norm = ||b||_2
    double b_norm = sqrt(dot(n*T,b,b));
    //r_0 = b - Ax_0
    //Ax_apply_stencil_forward_euler(n, T, deltaT, x0, r_0, L);
    Ax_apply_stencil_backward_euler(n, T, deltaT, x0, r_0, L);
    axpby(n*T,1.0,b,-1.0,r_0);
    block_cg_solver(n,T,deltaT,r_0,r_0,L);
    //Q[:,0] = r_0/||r_0||_2
    double r0_norm = sqrt(dot(n*T,r_0,r_0));
    vec2matrix(n*T,0,r_0,Q);
    matrix_col_scale(n*T, 0, r0_norm, Q);
    //e[0] = r0_norm
    e_1[0] = r0_norm;
    //arnoldi iteration
    for (int j = 0; j<maxIter; j++){
        precond_cg_arnoldi(n,T, maxIter, j,deltaT,epsilon,Q,H,L);
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
    backward_substitution(iter, maxIter, e_1,H, y);
    // x^{*} = Qy + x0 
    matrix_vec_prod(n*T, iter+2, x, Q, y);
    //vector_scale(n*T,perturb,x);
    axpby(n * T, 1.0, x0, 1.0, x);

    // r = b-Ax
    //Ax_apply_stencil_forward_euler(n, T, deltaT, x, Ax, L);
    Ax_apply_stencil_backward_euler(n, T, deltaT, x, Ax, L);
    //vector_scale(n*T,perturb,Ax);
    axpby(n*T, 1.0, b, -1.0, Ax);
    // rel_r_norm = ||r||_2/||b||_2 
    double rel_r_norm = sqrt(dot(n*T, Ax, Ax))/b_norm;
    std::cout <<"Relative residual: "<<rel_r_norm <<" (||r||_2/||r_0||_2)"<< std::endl;
    *resNorm = rel_r_norm;

    delete [] r_0;
}













