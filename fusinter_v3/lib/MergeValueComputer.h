#ifndef FUSINTER_V3_MERGEVALUECOMPUTER_H
#define FUSINTER_V3_MERGEVALUECOMPUTER_H

#include<Eigen/Dense>

#include<cmath>

#include "typedefs.h"

namespace lib {
    float shannon_entropy(
            Eigen::VectorXi input_column,
            float alpha,
            float lam,
            int m,
            int n
    ) {

        float col_sum = 0;
        int n_j = input_column.sum();
        float col_fac = alpha * n_j / n;

        for (int i = 0; i < m; i++) {
            float p = (input_column[i] + lam) / (n_j + m * lam);
            col_sum += -(p * std::log2(p));
        }
        return col_fac * col_sum + (1 - alpha) * m * lam / n_j;
    }

    class MergeValueComputer {
    private:
        lib::table table;
        float alpha;
        float lam;
        int m;
        int n;
        std::vector<float> cols_entropy;
        std::vector<float> deltas;

    public:
        MergeValueComputer(const lib::table &table, float alpha, float lam)
        :table(table),
        alpha(alpha),
        lam(lam)
        {
            this->m = table.rows();
            this->n = table.sum();

            for(int i; i < this->table.cols(); i++){
                auto col = this->table.col(i);
                auto entropy = shannon_entropy(col, this->alpha, this->lam, this->m, this->n);
                this->cols_entropy.push_back(entropy);
            }

        //TODO init deltas

        std::cout << get_table_entropy() << std::endl;
        };
    private:
        float compute_merge_entropy(int col_idx){
            auto col = this->table.block(0, col_idx, this->m, 2)
            .rowwise()
            .sum();

            return shannon_entropy(col, this->alpha, this->lam, this->m, this->n);
        };

        float get_table_entropy(){
            float result = 0;
            for(auto el: this->cols_entropy)
                result += el;
            return result;
        };

        float compute_delta(int col_idx, bool left=false){
            return -1;
        };

        std::vector<float> get_all_deltas(){
            return std::vector<float>{};
        }

        void update(lib::table table, int max_ind){
           return;
        }
    };
}

#endif
