#ifndef FUSINTER_V3_TABLEMANAGER_H
#define FUSINTER_V3_TABLEMANAGER_H

#include<vector>

#include <Eigen/Dense>

#include "typedefs.h"

class TableManager {
    data_vec data_x;
    label_vec data_y;

    TableManager(data_vec data_x, label_vec data_y) : data_x(data_x), data_y(data_y) {};

    table create_table(const std::vector<int> &init_splits) {
        return table{{1},
                     {2},
                     {3}};
    }
};

#endif //FUSINTER_V3_TABLEMANAGER_H
