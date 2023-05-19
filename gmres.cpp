#include "operations.hpp"
#include "gmres.hpp"
#include "time_integration.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

#include <iostream>
#include <iomanip>
#include <omp.h>

void gmres(stencil3d const* L, const double* b, double* x0, int const maxIter, double epsilon, int n ,double delta_t,  int T, double* resNorm) {
    int const maxIter_p1 = maxIter + 1;
    double Q[n * T * maxIter_p1] = {0.0};
    double H[maxIter_p1 * maxIter] = {0.0};
    double H_g[maxIter_p1 * maxIter] = {0.0};
    double e_1[maxIter_p1] = {0.0};
    double* sol = new double[n * T]();
    double* Asol = new double[n * T]();
    double* r = new double[n * T]();
    double* Ax = new double[n * T]();
    double* AQ = new double[n * T]();
    double* Q_j = new double[n * T]();
    double r_norm = 1.0;
    double res = 0.0;
    double denom = 0.0;
    double y[maxIter_p1] = {0.0}; // initialize y to 0
    double c = 0.0;
    double s = 0.0;
    int iter;

    // Ax = A*x0
    //Ax_apply_stencil(L, x0, Ax, T, n, delta_t);
    apply_stencil3d(L,x0,Ax);
    // r_0 (= Ax) = b - Ax
    axpby(n * T, 1.0, b, -1.0, Ax);
    r_norm = sqrt(dot(n * T, Ax, Ax));
    // Q[:][0] = r_0/||r_0||_2
    for (int i = 0; i < n * T; i++) {
        Q[index(i, 0, n * T)] = Ax[i] / r_norm;
    }
    // Set e_1 to be [beta,0,...0]
    init(maxIter_p1, e_1, 0.0);
    e_1[0] = r_norm;
    
    // Perform the Arnoldi iteration
    for (int j = 0; j < maxIter; j++) {
        std::cout << "Iteration: " << j << std::endl;
        // Put Q[:,j] into Q_j
        for (int i = 0; i < n * T; i++) {
            Q_j[i] = Q[index(i, j, n * T)];
        }
        // Calculate A*Q[:,j]
        //Ax_apply_stencil(L, Q_j, AQ, T, n, delta_t);
        apply_stencil3d(L,Q_j,AQ);
        // Put AQ into Q[:,j+1]
        for (int i = 0; i < n * T; i++) {
            Q[index(i, j + 1, n * T)] = AQ[i];
        }
        for (int i = 0; i < j + 1; i++) {
            // H[i][j] = Q[:][i]^T*Q[:,j+1]
            H[index(i, j, maxIter_p1)] = 0.0;
            for (int k = 0; k < n * T; k++) {
                H[index(i, j, maxIter_p1)] += Q[index(k, i, n * T)] * AQ[k];
            }
            // Q[:][j+1] = Q[:][j+1] - H[i][j]*Q[:,i]
            for (int k = 0; k < n * T; k++) {
                Q[index(k, j + 1, n * T)] -= H[index(i, j, maxIter_p1)] * Q[index(k, i, n * T)];
            }
        }
        // H[j+1][j] = ||Q[:][j+1]||_2
        H[index(j + 1, j, maxIter_p1)] = 0.0;
        for (int k = 0; k < n * T; k++) {
            H[index(j + 1, j, maxIter_p1)] += Q[index(k, j + 1, n * T)] * Q[index(k, j + 1, n * T)];
        }
        H[index(j + 1, j, maxIter_p1)] = sqrt(H[index(j + 1, j, maxIter_p1)]);
        // Q[:][j+1] = Q[:][j+1]/H[j+1][j]
        for (int k = 0; k < n * T; k++) {
            Q[index(k, j + 1, n * T)] /= H[index(j + 1, j, maxIter_p1)];
        }
    
        // Givens rotation on H_:j+2,:j+1 to make upper triangular matrix = R
        denom = sqrt(H[index(j, j, maxIter_p1)] * H[index(j, j, maxIter_p1)] + H[index(j + 1, j, maxIter_p1)] * H[index(j + 1, j, maxIter_p1)]);
        c = H[index(j, j, maxIter_p1)] / denom;
        s = H[index(j + 1, j, maxIter_p1)] / denom;
        
        for (int i = 0; i < maxIter; i++) {
            H_g[index(j, i, maxIter_p1)] = c * H[index(j, i, maxIter_p1)] + s * H[index(j + 1, i, maxIter_p1)];
            H_g[index(j + 1, i, maxIter_p1)] = -s * H[index(j, i, maxIter_p1)] + c * H[index(j + 1, i, maxIter_p1)];
        }
        for (int i = 0; i < maxIter; i++) {
            H[index(j, i, maxIter_p1)] = H_g[index(j, i, maxIter_p1)];
            H[index(j+1, i, maxIter_p1)] = H_g[index(j+1, i, maxIter_p1)];
        }
        e_1[j + 1] = -s * e_1[j];
        e_1[j] = c * e_1[j];
        H[index(j+1, j, maxIter_p1)] = 0.0;

        if ((std::abs(e_1[j + 1]) < epsilon) || (iter==maxIter)){
            iter = j;
            std::cout <<"Iterations Stopped"<< std::endl;
            std::cout <<"Iter:"<<iter << std::endl;
            std::cout <<"Epsilon:"<<epsilon << std::endl;
            std::cout <<"abs(e_1[j+1]):"<<std::abs(e_1[j + 1]) << std::endl;
            break;
        }
    }

    y[iter] = e_1[iter]/H[index(iter, iter, maxIter_p1)];
    for (int i=(iter-1); i>=0; i--){
        y[i] = e_1[i];        
        for (int j=i+1; j<iter+2; j++){
            y[i] -= H[index(i, j, maxIter_p1)]*y[j];
        }           
        y[i] = y[i] / H[index(i, i, maxIter_p1)];
    }

    std::cout << "y minnorm sol ";
    for (int i = 0; i < maxIter_p1; i++) {
        std::cout << y[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < n * T; i++) {
        sol[i] = 0.0;
        for (int k = 0; k < iter ; k++) {
            sol[i] += Q[index(i, k, maxIter_p1)] * y[k];
        }
        //std::cout << "sol["<<i<<"]:"<<sol[i]<<std::endl;
    }
    axpby(n*T, 1.0, x0, 1.0, sol);
    //Ax_apply_stencil(L, y, sol, T, n, delta_t);
    apply_stencil3d(L,sol,Asol);
    
    axpby(n*T, 1.0, b, -1.0, Asol);
    for (int i = 0; i < n * T; i++) {
        res +=  Asol[i]*Asol[i];
    }
    res = sqrt(res);
    *resNorm = res;
    
}
