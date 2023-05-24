#include <cmath>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include "operations.hpp"
#include "timer.hpp"
#include "gmres_method.hpp"


void gmres(stencil3d const* A, const double* b, double* x, int const maxIter, double epsilon, double* resNorm){
    int n = sizeof(b);
    int const maxIter_p1 = maxIter + 1;
    double *Q = new double[n*maxIter_p1];
    double H[maxIter_p1*maxIter] = {0.0};
    double e_1[maxIter_p1] = {0.0};
    double *r = new double[n];
    double *Ax = new double[n];
    double *AQ = new double[n];
    double *Q_j = new double[n];
    double r_norm = 1.0;
    double res = 1.0;
    
    // Set e_1 to be [1,0,...0]
    e_1[0] = 1.0;

    printf("Gestart");
    apply_stencil3d(A, x, Ax);

    // Ax (= r_0) = b - Ax
    axpby(n, 1.0, b, -1.0, Ax);
    r_norm = sqrt(dot(n, Ax, Ax));
    // Q[:][0] = r_0/||r_0||^2
    for(int i=0; i<n; n++){
           Q[index(i, 0, maxIter_p1)]= Ax[i]/r_norm;
        }
    // Perform the Arnoldi iteration
    for(int j=0; j<maxIter; j++){
        printf("Iteratie gestart");
        // Calculate A*Q[:,j] and put it into Q[:, j+1]
        // Put Q[:,j] into Q_j and Q[:,]
        for (int i=0; i<n; i++){
            Q_j[i] = Q[index(i,j,maxIter_p1)];
        }
        apply_stencil3d(A, Q_j, AQ);
        // Put AQ into Q[:,j+1]
        for (int i=0; i<n; i++){
            Q[index(i,j+1,maxIter_p1)] = AQ[i];
        }
        
        // TODO misschien twee losse for loops
        for (int i=0; i<j; i++){
            // H[i][j] = Q[:][i]^T*Q[:,j+1]
            for (int k=0;k<n;k++){
                H[index(i,j,maxIter)] += Q[index(k,i,maxIter_p1)]*Q[index(k,j+1,maxIter_p1)];
            }
            // Q[:][j+1] = Q[:][j+1] - H[i][j]*Q[:,i]
            for (int k=0;k<n;k++){
                Q[index(k,j+1,maxIter_p1)] = Q[index(k,j+1,maxIter_p1)] - H[index(i,j,maxIter)]*Q[index(k,i,maxIter_p1)];
            }
        }

        // H[j+1][j] = norm(Q[:][j+1])
        for (int k=0;k<n;k++){
            H[index(j+1,j,maxIter)] = Q[index(k,j+1,maxIter_p1)]*Q[index(k,j+1,maxIter_p1)];
        }
        H[index(j+1,j,maxIter)] = sqrt(H[index(j+1,j,maxIter)]);

        // Avoid dividing by zero
        if (abs(H[index(j+1,j,maxIter)]) > epsilon) {
            // Q[:][j+1] = Q[:][j+1]/H[j+1][j]
            for (int k=0;k<n;k++){
                Q[index(k,j+1,maxIter_p1)] = Q[index(k,j+1,maxIter_p1)]/H[index(j+1,j,maxIter)];
            }
        }
        
        // Solve for y: H[:j+2][:j+1]*y = beta*e_1
        for (int i=0; i<maxIter+1; i++){
            for (int k=0; k<maxIter; k++){
                std::cout << H[index(i,j,maxIter)] << " ";
            }
            std::cout << std::endl;
        }
    //     y = ...


    //     res = dot(n,)
    //     if (res < epsilon){
    //         for (int i=0;i<n;i++){
    //             for (int k=0;k<n;k++){
    //                 sol[i] = Q[i][k]*Y[k] + x0[i]
    //             }
    //         }
    //         return sol, res
    //     }

    }
    // for (int i=0;i<n;i++){
    //     for (int k=0;k<n;k++){
    //         sol[i] = Q[i][k]*Y[k] + x0[i]
    //     }
    // }
    *resNorm = res;

    delete [] Q;
    delete [] r;
    delete [] Ax;
    delete [] AQ;
    delete [] Q_j;

    return;
}