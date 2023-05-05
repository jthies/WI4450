#include <cmath>
#include "operations.hpp"
#include "timer.hpp"
#include "GMRES_method.hpp"

void GMRES(double* A, double*b,double const* x0, int const maxIter, double epsilon){
    int n = sizeof(b);
    double *Q = new double[n][k+1];
    double *H[k+1][k] = {0.0};
    double *r = new double[n];
    double *Ax = new double[n];
    double r_norm = 1.0;
    double res = 1.0;
    double* sol = new double[n];
    
    for (int i=0; i<n; i++){
      for (int j=0; j<n; j++){
        Ax[i]+= A[i][j]*x0[j];
      }
    }
    axpby(n, 1.0, b, -1.0, double* Ax)
    r_norm = dot(n,Ax,Ax)
    for(int i=0; i<n; n++){
           Q[i][0]= Ax[i]/r_norm;
        }
    for(int j=0; j<maxIter; j++){
        for (int i=0; i<n; i++){
            for (int k=0; k<n; k++){
                Q[i][j+1]+= A[i][k]*Q[k][j]
            }
        }
        for (int i=0; i<n; i++){
            for (int k=0;k<n;k++){
                H[i][j] = Q[k][i]*Q[k][j+1]
                Q[k][j+1] = Q[k][j+1] - H[i][j]*Q[k][i]
            }
            for (int k=0;k<n;k++){
                H[j+1][j] = Q[k][j+1]*Q[k][j+1]
            }
            if (abs(H[j+1][j]) >epsilon ) {
                for (int k=0;k<n;k++){
                    Q[k][j+1] = Q[k][j+1]/H[j+1][j];
                }
            }
        }
        y = ...
        res = dot(n,)
        if (res < epsilon){
            for (int i=0;i<n;i++){
                for (int k=0;k<n;k++){
                    sol[i] = Q[i][k]*Y[k] + x0[i]
                }
            }
            return sol, res
        }
    }
    for (int i=0;i<n;i++){
        for (int k=0;k<n;k++){
            sol[i] = Q[i][k]*Y[k] + x0[i]
        }
    }
    return sol, res
}