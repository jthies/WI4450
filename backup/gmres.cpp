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
    double b_norm;
    double res = 0.0;
    double denom = 0.0;
    double y[maxIter_p1] = {0.0}; // initialize y to 0
    double c = 0.0;
    double s = 0.0;
    int iter;

    // Ax = A*x0 --> NOTE: with x0 = 0, Ax = 0 and this computation is not necessary 
    Ax_apply_stencil(L, x0, Ax, T, n, delta_t);
    
    // r_0 (=: Ax) = b - Ax
    axpby(n * T, 1.0, b, -1.0, Ax);
    r_norm = sqrt(dot(n * T, Ax, Ax));
    b_norm = sqrt(dot(n * T, b, b));

    // Q[:][0] = r_0/||r_0||_2
    for (int i = 0; i < n * T; i++) {
        Q[index(i, 0, n * T)] = Ax[i] / r_norm;
    }
    // Set e_1 to be [beta,0,...0]
    init(maxIter_p1, e_1, 0.0);
    e_1[0] = r_norm;
    
    // Perform the Arnoldi iteration
    for (int j = 0; j < maxIter; j++) {
        // Put Q[:,j] into Q_j
        for (int i = 0; i < n * T; i++) {
            Q_j[i] = Q[index(i, j, n * T)];
        }
        // Calculate A*Q[:,j]
        Ax_apply_stencil(L, Q_j, AQ, T, n, delta_t);
        //apply_stencil3d(L,Q_j,AQ);
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
        
        if (H[index(j + 1, j, maxIter_p1)] > epsilon*1e-5){
            // Q[:][j+1] = Q[:][j+1]/H[j+1][j]
            for (int k = 0; k < n * T; k++) {
                Q[index(k, j + 1, n * T)] /= H[index(j + 1, j, maxIter_p1)];
            }
        } else {
            std::cout << "Stopped since H[j+1,j] < epsilon" << std::endl;
        }

        // Remember the e_1 and H as they are original/
        double H_origin[maxIter_p1 * maxIter] = {0.0};
        double e_1_origin[maxIter_p1] = {0.0}; 
        for(int l = 0; l < maxIter_p1 * maxIter; l++){
            H_origin[l] = H[l];
        }
        e_1_origin[0] = r_norm;
    
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

        // //Print e_1 with the Givens rotation
        // std::cout << "e_1 with Givens rotation" << std::endl;
        // for (int i = 0; i < maxIter; i++) {
        //     std::cout << e_1[i] << " ";
        // }

        // // //Print H with the Givens rotation
        // std::cout << "H with Givens rotation" << std::endl;
        // for (int i = 0; i < maxIter_p1; i++) {
        //     for (int k = 0; k < maxIter; k++) {
        //         std::cout << H[index(i, k, maxIter_p1)] << " ";
        //     }
        //     std::cout << std::endl;
        // }

        std::cout << "error iteration " << j << " is " << std::abs(e_1[j])/b_norm << std::endl;

        // if (iter==maxIter){
        if ((std::abs(e_1[j+1])/b_norm < epsilon) || (iter==maxIter)){
            iter = j;
            std::cout <<"Iterations Stopped"<< std::endl;
            std::cout <<"Iter:"<<iter << std::endl;
            std::cout <<"Epsilon:"<<epsilon << std::endl;
            std::cout <<"abs(e_1[j+1]):"<<std::abs(e_1[j+1]) << std::endl;
            break;
        }
    }

    // //Print e_1 with the Givens rotation
    // std::cout << "e_1 with Givens rotation" << std::endl;
    // for (int i = 0; i < maxIter; i++) {
    //     std::cout << e_1[i] << " ";
    // }
    // std::cout << std::endl;

    // // //Print H with the Givens rotation
    // std::cout << "H with Givens rotation" << std::endl;
    // for (int i = 0; i < maxIter_p1; i++) {
    //     for (int k = 0; k < maxIter; k++) {
    //         std::cout << H[index(i, k, maxIter_p1)] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // Back substitution
    y[iter] = e_1[iter]/H[index(iter, iter, maxIter_p1)];
    for (int i=(iter-1); i>=0; i--){
        y[i] = e_1[i];        
        for (int j=i+1; j <= iter; j++){
            y[i] -= H[index(i, j, maxIter_p1)]*y[j];
        }           
        y[i] = y[i] / H[index(i, i, maxIter_p1)];
    }

    // Print the solution for the least squares problem.
    std::cout << "y minnorm sol ";
    for (int i = 0; i < maxIter_p1; i++) {
        std::cout << y[i] << " ";
    }
    std::cout << std::endl;

    // // Print Q[:,:j+1]
    // std::cout << "Q[:,:j+1]" << std::endl;
    // for (int i = 0; i < n*T; i++) {
    //     for (int k = 0; k < iter + 1; k++) {
    //         std::cout << Q[index(i, k, maxIter_p1)] << " ";
    //     }
    //     std::cout << std::endl;
    // }


    // Actual solution Qy calculation with Q
    init(n * T, sol, 0.0);
    for (int i = 0; i < n * T; i++) {
        for (int k = 0; k <= iter + 1; k++) {
            sol[i] += Q[index(i, k, maxIter_p1)] * y[k];
        }
        // std::cout << "sol["<<i<<"]:"<<sol[i]<<std::endl;
    }
    // sol = Qy + x0
    axpby(n * T, 1.0, x0, 1.0, sol);

    // Calculate residual b - A*sol
    init(n * T, Asol, 0);
    Ax_apply_stencil(L, sol, Asol, T, n, delta_t);
    axpby(n*T, 1.0, b, -1.0, Asol);

    // Calculate residual norm
    res = sqrt(dot(n*T, Asol, Asol));
    // res = sqrt(res)/r_norm;
    std::cout << "residual (from b-Ax):"<< res <<std::endl;
    std::cout << "residual (from e_1[j+1]):"<< e_1[iter+1] <<std::endl;
    *resNorm = res;
    
}