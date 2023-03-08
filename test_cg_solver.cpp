#include "gtest_mpi.hpp"

#include "operations.hpp"
#include "cg_solver.cpp"

#include <iostream>
#include <cmath>

TEST(cg_solver, poisson){
    const int nx=4, ny=4, nz=4; // define dimensions
    const int n=nx*ny*nz;       // define length vector
    double* rhs=new double[n];  // define right hand side vector
    double* u_known=new double[n];    // define solution vector
    double* u=new double[n];    // define solution vector

    for (int i=0;i<n;i++){      
        u_known[i] = i%2; // solution is known
        u[i] = 1.0;             // starting vector
    }

    stencil3d S;

    S.nx=nx; S.ny=ny; S.nz=nz;
    S.value_c = 6;
    S.value_n = -1;
    S.value_e = -1;
    S.value_s = -1;
    S.value_w = -1;
    S.value_b = -1;
    S.value_t = -1;

    apply_stencil3d(&S,u_known,rhs);

    double cg_tol = 10e-60;
    double tol = 10e-6;
    int maxIter = 1000;
    int numIter = 0;
    double resNorm = 10e6;

    cg_solver(&S, n, u, rhs, cg_tol, maxIter, &resNorm, &numIter,0);

    double err=0.0;

    for (int i=0; i<n; i++) err = std::max(err, std::abs(u[i]-u_known[i]));

    EXPECT_NEAR(1.0+err, 1.0, tol);
    
    delete [] rhs;
    delete [] u_known;
    delete [] u;
}