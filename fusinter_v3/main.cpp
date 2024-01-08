//
// Created by floxo on 12/28/23.
//

#include "paper_data.h"
#include <iostream>

#include "lib/library.h"
int main(){
    auto splitter = lib::Splitter(paper_data_x, paper_data_y);
    auto splits = splitter.apply();

//    for (auto el: splits){
//        std::cout << el << ", ";
//    }

//    std::cout << std::endl;


    auto tm = lib::TableManager(paper_data_x, paper_data_y);
    auto table = tm.create_table(splits);
    std::cout << table << std::endl;
}