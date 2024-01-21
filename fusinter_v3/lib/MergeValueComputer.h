#include<Eigen/Dense>
#include<cmath>
float shannon_entropy(
        Eigen::VectorXi input_column,
        float alpha,
        float lam,
        int m,
        int n
        ){

    float col_sum = 0;
    int n_j = input_column.sum();
    float col_fac = alpha * n_j / n;

    for(int i = 0; i < m; i++){
        float p = (input_column[i] + lam) / (n_j + m * lam);
        col_sum += -(p* std::log2(p));
    }
    return col_fac * col_sum + (1-alpha) * m * lam / n_j;
}