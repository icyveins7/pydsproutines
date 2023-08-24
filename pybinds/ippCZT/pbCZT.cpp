#include "CZT.h"

namespace py = pybind11;

// This module name in PYBIND11_MODULE must be the same as the filename!
// https://github.com/pybind/pybind11/blob/master/docs/faq.rst
PYBIND11_MODULE(pbIppCZT32fc, m) {
    py::class_<IppCZT32fc>(m, "pbIppCZT32fc")
        .def(py::init<int, Ipp32f, Ipp32f, Ipp32f, Ipp32f>())
        .def("run", &IppCZT32fc::run)
        .def_readonly("m_k", &IppCZT32fc::m_k)
        .def_readonly("m_N", &IppCZT32fc::m_N)
        // .def_readonly("m_ww", &IppCZT32fc::m_ww)
        // .def_readonly("m_fv", &IppCZT32fc::m_fv)
        // .def_readonly("m_aa", &IppCZT32fc::m_aa)
        ;

    m.doc() = "pybind11 for IppCZT32fc"; // optional module docstring
}