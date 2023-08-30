#include "GroupXcorrCZT.h"

namespace py = pybind11;

// This module name in PYBIND11_MODULE must be the same as the filename!
// https://github.com/pybind/pybind11/blob/master/docs/faq.rst
PYBIND11_MODULE(pbIppGroupXcorrCZT, m) {
    py::class_<GroupXcorrCZT>(m, "pbIppGroupXcorrCZT")
        .def(py::init<int, double, double, double, double, size_t>())
        // .def("run", &GroupXcorrCZT::run,
        //     "Examples:\n"
        //     ".run(x) # x is a 1d numpy array\n"
        //     ".run(x[i,:]) # x is a 2d numpy array, operating on 1 row (see runMany)"
        // )
        // .def("runMany", &IppCZT32fc::runMany,
        //     "Example:\n"
        //     ".runMany(x) # x is a 2d numpy array, operating on every row\n"
        // )
        // .def_readonly("m_k", &IppCZT32fc::m_k)
        // .def_readonly("m_N", &IppCZT32fc::m_N)
        ;

    m.doc() = "pybind11 for GroupXcorrCZT IPP implementation"; // optional module docstring
}