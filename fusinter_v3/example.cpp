#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <iostream>

int add(int i, int j) {
    return i + j;
}

Eigen::Matrix<int, -1, -1> & test_np_to_eigen(Eigen::Matrix<int, -1, -1> &m){
    std::cout << m << std::endl;
    return m;
}

PYBIND11_MODULE(EXAMPLE, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function that adds two numbers");
    m.def("test_np_to_eigen", &test_np_to_eigen, "test if I can convert np to eigen");
}