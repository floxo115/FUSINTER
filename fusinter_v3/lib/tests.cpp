#define CATCH_CONFIG_MAIN

#include<tuple>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "../paper_data.h"
#include "errors.h"
#include "Splitter.h"
#include "TableManager.h"
#include "typedefs.h"

/////////////////////////////////////////////////////// SPLITTER TESTS /////////////////////////////////////////////////

TEST_CASE("Splitter with unsorted input") {
    auto data_y = paper_data_y;
    auto data_x = paper_data_x;

    data_x[0] = data_x[70];
    SECTION("should throw NOT_SORTED_ERROR") {
        REQUIRE_THROWS_AS(lib::Splitter(data_x, data_y), lib::NOT_SORTED_ERROR);
    }
}

TEST_CASE("Splitter with non matching labels and data vectors") {
    auto data_y = paper_data_y(Eigen::seq(0, 80));
    auto data_x = paper_data_x;
    SECTION("should throw NOT_MATCHING_DATA_SIZES") {
        REQUIRE_THROWS_AS(lib::Splitter(data_x, data_y), lib::NOT_MATCHING_DATA_SIZES);
    }
}

struct splitter_test_data {
    lib::data_vec data_x;
    lib::label_vec data_y;
    std::vector<float> expected;
};

TEST_CASE("Splitter with different valid inputs") {
    auto data = GENERATE(
            splitter_test_data(
                    paper_data_x,
                    paper_data_y,
                    {2, 3, 13, 14, 15, 16, 17, 18, 19, 20, 23, 37, 38, 39, 40}
            ),
            splitter_test_data(
                    lib::data_vec{{-10, -10, -10., -9., -9., -8., -8., -8., -8., 2., 2., 3., 3.}},
                    lib::label_vec{{0, 0, 0, 0, 0, 0, 1, 1, 0, 2, 2, 2, 2}},
                    {-8., 2.}
            )
    );
    auto splitter = lib::Splitter(data.data_x, data.data_y);
    REQUIRE(splitter.apply() == data.expected);
}

/////////////////////////////////////////////////// TABLE MANAGER TESTS ///////////////////////////////////////////////

TEST_CASE("TableManager with unsorted input") {
    auto data_y = paper_data_y;
    auto data_x = paper_data_x;

    data_x[0] = data_x[70];
    SECTION("should throw NOT_SORTED_ERROR") {
        REQUIRE_THROWS_AS(lib::TableManager(data_x, data_y), lib::NOT_SORTED_ERROR);
    }
}

TEST_CASE("TableManager with unequal sized input") {
    auto data_y = lib::label_vec{{0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0}};
    auto data_x = paper_data_x;

    SECTION("should throw NOT_SORTED_ERROR") {
        REQUIRE_THROWS_AS(lib::TableManager(data_x, data_y), lib::NOT_MATCHING_DATA_SIZES);
    }
}

struct table_manager_test_input {
    lib::data_vec data_x;
    lib::label_vec data_y;
    std::vector<float> input_splits;
    lib::table expected;
};

TEST_CASE("TableManager with valid inputs") {
    auto data = GENERATE(
            table_manager_test_input(
                    paper_data_x,
                    paper_data_y,
                    {2, 3, 13, 14, 15, 16, 17, 18, 19, 20, 23, 37, 38, 39, 40},
                    lib::table{{1, 0, 0, 1, 26, 0, 2, 1, 3, 0, 0, 3, 3, 0, 0, 3, 2, 1, 0, 2, 5, 0, 0, 27, 1,
                                2, 2, 1, 0, 2, 2, 0}}.reshaped(2, 16)
            ),
            table_manager_test_input(
                    lib::data_vec {{-10, -10, -10, -9, -9, -8, -8, -8, -8, 2, 2, 3, 3}},
                    lib::label_vec {{0, 0, 0, 0, 0, 0, 1, 1, 0, 2, 2, 2, 2}},
                    {-8., 2.},
                    lib::table{{5, 0, 0, 2, 2, 0, 0, 0, 4}}.reshaped(3, 3)
            )
    );

    auto tm = lib::TableManager(data.data_x, data.data_y);
    REQUIRE(tm.create_table(data.input_splits) == data.expected);
}