#include <catch2/catch_test_macros.hpp>

#include "../paper_data.h"
#include "errors.h"
#include "Splitter.h"

//TEST_CASE( "Factorials are computed", "[factorial]" ) {
//    REQUIRE( factorial( 1) == 1 );
//    REQUIRE( factorial( 2) == 2 );
//    REQUIRE( factorial( 3) == 6 );
//    REQUIRE( factorial(10) == 3'628'800 );
//}

TEST_CASE("Splitter with unsorted input"){
    auto data_y = paper_data_y;
    auto data_x = paper_data_x;

    data_x[0] = data_x[70];
    SECTION("should throw NOT_SORTED_ERROR") {
        REQUIRE_THROWS_AS(lib::Splitter(data_x, data_y), lib::NOT_SORTED_ERROR);
    }
}

TEST_CASE("Splitter with non matching labels and data vectors"){
    auto data_y = paper_data_y(Eigen::seq(0, 80));
    auto data_x = paper_data_x;
    SECTION("should throw NOT_MATCHING_DATA_SIZES") {
        REQUIRE_THROWS_AS(lib::Splitter(data_x, data_y), lib::NOT_MATCHING_DATA_SIZES);
    }

}