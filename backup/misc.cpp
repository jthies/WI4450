// Check if Q_i with Q_j+1 are orthonormal
        for (int i = 0; i <= j + 1; i++) {
            double nor = 0.0;
            for (int k = 0; k < n * T; k++) {
                nor += Q[index(k, i, n * T)] * Q[index(k, j + 1, n * T)];
            }
            std::cout << "Norm: " << nor << std::endl;
        }
        std::cout << std::endl;
        std::cout << "H without Givens rotation:" << std::endl;
        for (int i = 0; i < maxIter_p1; i++) {
            for (int k = 0; k < maxIter; k++) {
                std::cout << H[index(i, k, maxIter_p1)] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        std::cout << "e_1 without Givens rotation:" << std::endl;
        for (int i = 0; i < j + 1; i++) {
            std::cout << e_1[i] << " ";
        }
        std::cout << std::endl;
        std::cout << std::endl;

        // Givens rotation on H_:j+2,:j+1 to make upper triangular matrix = R
        //denom = sqrt(H[index(j, j, maxIter_p1)] * H[index(j, j, maxIter_p1)] + H[index(j + 1, j, maxIter_p1)] * H[index(j + 1, j, maxIter_p1)]);
        //c = H[index(j, j, maxIter_p1)] / denom;
        //s = -H[index(j + 1, j, maxIter_p1)] / denom;
        //for (int i = 0; i < maxIter; i++) {
        //    H_g[index(j, i, maxIter_p1)] = c * H[index(j, i, maxIter_p1)] + s * H[index(j + 1, i, maxIter_p1)];
        //    H_g[index(j + 1, i, maxIter_p1)] = -s * H[index(j, i, maxIter_p1)] + c * H[index(j + 1, i, maxIter_p1)];
        //}
        //e_1_g[j] = c * e_1[j] + s * e_1[j + 1];
        //e_1_g[j + 1] = -s * e_1[j] + c * e_1[j + 1];
        //e_1[j] = e_1_g[j];
        //e_1[j + 1] = e_1_g[j + 1];
        //for (int i = 0; i <= j; i++) {
        //    H[index(i, j, maxIter_p1)] = H_g[index(i, j, maxIter_p1)];
        //    H[index(i, j + 1, maxIter_p1)] = H_g[index(i, j + 1, maxIter_p1)];
        //}
        
        // Givens rotation on H_:j+2,:j+1 to make upper triangular matrix = R
        denom = sqrt(H[index(j, j, maxIter_p1)] * H[index(j, j, maxIter_p1)] + H[index(j + 1, j, maxIter_p1)] * H[index(j + 1, j, maxIter_p1)]);
        c = H[index(j, j, maxIter_p1)] / denom;
        s = -H[index(j + 1, j, maxIter_p1)] / denom;
        for (int i = 0; i < maxIter; i++) {
            H_g[index(j, i, maxIter_p1)] = c * H[index(j, i, maxIter_p1)] + s * H[index(j + 1, i, maxIter_p1)];
            H_g[index(j + 1, i, maxIter_p1)] = -s * H[index(j, i, maxIter_p1)] + c * H[index(j + 1, i, maxIter_p1)];
        }
        e_1_g[j] = c * e_1[j] + s * e_1[j + 1];
        e_1_g[j + 1] = -s * e_1[j] + c * e_1[j + 1];
        e_1[j] = e_1_g[j];
        e_1[j + 1] = e_1_g[j + 1];
        for (int i = 0; i <= j; i++) {
            H[index(i, j, maxIter_p1)] = H_g[index(i, j, maxIter_p1)];
            H[index(i, j + 1, maxIter_p1)] = H_g[index(i, j + 1, maxIter_p1)];
        }

        std::cout << "H with Givens rotation" << std::endl;
        for (int i = 0; i < maxIter + 1; i++) {
            for (int k = 0; k < maxIter; k++) {
                std::cout << H[index(i, k, maxIter_p1)] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        std::cout << "e_1 with Givens rotation" << std::endl;
        for (int i = 0; i < maxIter; i++) {
            std::cout << e_1[i] << " ";
        }
        std::cout << std::endl;
        std::cout << std::endl;

        std::cout << "H with Givens rotation" << std::endl;
    for (int i = 0; i < maxIter + 1; i++) {
        std::cout << i<<":";
        for (int k = 0; k < maxIter; k++) {
            std::cout << H[index(i, k, maxIter_p1)] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;