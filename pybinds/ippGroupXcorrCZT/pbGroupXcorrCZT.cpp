#include "GroupXcorrCZT.h"

namespace py = pybind11;

// This module name in PYBIND11_MODULE must be the same as the filename!
// https://github.com/pybind/pybind11/blob/master/docs/faq.rst
PYBIND11_MODULE(pbIppGroupXcorrCZT, m) {
    py::class_<GroupXcorrCZT>(m, "pbIppGroupXcorrCZT")
        .def(py::init<int, double, double, double, double>())
        .def(py::init<int, double, double, double, double, size_t>()) // default args don't seem to work for constructors..
        .def("getNumThreads", &GroupXcorrCZT::getNumThreads)
        .def("addGroup", &GroupXcorrCZT::addGroup)
        .def("resetGroups", &GroupXcorrCZT::resetGroups)
        .def("xcorr", &GroupXcorrCZT::xcorr)
        // .def("run", &GroupXcorrCZT::run,
        //     "Examples:\n"
        //     ".run(x) # x is a 1d numpy array\n"
        //     ".run(x[i,:]) # x is a 2d numpy array, operating on 1 row (see runMany)"
        // )
        ;

    m.doc() = "pybind11 for GroupXcorrCZT IPP implementation"; // optional module docstring
}