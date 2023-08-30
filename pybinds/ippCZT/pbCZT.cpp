#include "CZT.h"

namespace py = pybind11;

// This module name in PYBIND11_MODULE must be the same as the filename!
// https://github.com/pybind/pybind11/blob/master/docs/faq.rst
PYBIND11_MODULE(pbIppCZT32fc, m) {
    py::class_<IppCZT32fc>(m, "pbIppCZT32fc")
        .def(py::init<int, double, double, double, double>())
        .def("run", &IppCZT32fc::run,
            "Examples:\n"
            ".run(x) # x is a 1d numpy array\n"
            ".run(x[i,:]) # x is a 2d numpy array, operating on 1 row (see runMany)"
        )
        .def("runMany", &IppCZT32fc::runMany,
            "Example:\n"
            ".runMany(x) # x is a 2d numpy array, operating on every row\n"
        )
        .def_readonly("m_k", &IppCZT32fc::m_k)
        .def_readonly("m_N", &IppCZT32fc::m_N)
        // .def_readonly("m_ww", &IppCZT32fc::m_ww)
        // .def_readonly("m_fv", &IppCZT32fc::m_fv)
        // .def_readonly("m_aa", &IppCZT32fc::m_aa)
        ;

    m.doc() = "pybind11 for IppCZT32fc"; // optional module docstring
}