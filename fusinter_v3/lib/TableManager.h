#ifndef FUSINTER_V3_TABLEMANAGER_H
#define FUSINTER_V3_TABLEMANAGER_H

#include<vector>
#include<set>

#include <Eigen/Dense>

#include "typedefs.h"
#include "errors.h"
namespace lib {
    class TableManager {
    private:
        lib::data_vec data_x;
        lib::label_vec data_y;

    public:
        TableManager(const lib::data_vec &data_x, const lib::label_vec &data_y) : data_x(data_x), data_y(data_y) {
            if (!std::is_sorted(data_x.begin(), data_x.end())) {
                throw lib::NOT_SORTED_ERROR();
            }
            if (this->data_x.size() != this->data_y.size()) {
                throw lib::NOT_MATCHING_DATA_SIZES();
            }
        }

        lib::table create_table(const std::vector<float> &init_splits) {
            auto n_labels = std::set<int>{this->data_y.begin(), this->data_y.end()}.size();
            auto n_splits = init_splits.size() + 1;

            Eigen::Matrix<int, -1, -1> table(n_labels, n_splits);
            table.setZero();

            Eigen::VectorXi n_labels_in_interval(n_labels);
            n_labels_in_interval.setZero();

            auto i = 0;
            for (auto split_idx = 0; split_idx < init_splits.size(); split_idx++){
                auto split_val = init_splits[split_idx];
                while (this->data_x[i] < split_val) {
                    n_labels_in_interval[this->data_y[i]] += 1;
                    i += 1;
                }

                table.col(split_idx) = n_labels_in_interval;
                n_labels_in_interval.setZero();
            }

            while (i < this->data_x.size()){
                n_labels_in_interval[this->data_y[i]] += 1;
                i += 1;
            }
            table.col(init_splits.size()) = n_labels_in_interval;

            return table;
        }

        lib::table compress_table(const lib::table &input_table, const int i) {
            return Eigen::Matrix<int, -1, -1>{{1}};
        }
    };
}
#endif //FUSINTER_V3_TABLEMANAGER_H
