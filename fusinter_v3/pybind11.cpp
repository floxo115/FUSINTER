#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <iostream>
#include "lib/library.h"

PYBIND11_MODULE(FUSINTER_v3_pybind, m) {
    pybind11::class_<lib::FUSINTERDiscretizer>(m, "FUSINTERDiscretizer")
            .def(pybind11::init<float, float>())
            .def("fit", &lib::FUSINTERDiscretizer::fit);
            //.def("transform", &lib::FUSINTERDiscretizer::transform)
}
