#include "GroupXcorrCZT.h"

namespace py = pybind11;

// This module name in PYBIND11_MODULE must be the same as the filename!
// https://github.com/pybind/pybind11/blob/master/docs/faq.rst
PYBIND11_MODULE(pbIppGroupXcorrCZT, m) {
    py::class_<GroupXcorrCZT>(m, "pbIppGroupXcorrCZT")
        .def(py::init<int, double, double, double, double>())
        .def(py::init<int, double, double, double, double, size_t>()) // default args don't seem to work for constructors..
        .def("getNumThreads", &GroupXcorrCZT::getNumThreads)
        .def("addGroup", 
            static_cast<void (GroupXcorrCZT::*)(int, const py::array_t<std::complex<float>, py::array::c_style>&, bool)>(&GroupXcorrCZT::addGroup),
            "Add a new group to the cross-correlation calculation.",
            py::arg("start"),
            py::arg("group"),
            py::arg("autoConj") = true
        )
        .def("addGroupsFromArray",
            static_cast<void (GroupXcorrCZT::*)(
                const py::array_t<int, py::array::c_style>&, 
                const py::array_t<int, py::array::c_style>&, 
                const py::array_t<std::complex<float>, py::array::c_style>&, 
                bool)>(&GroupXcorrCZT::addGroupsFromArray),
            "Adds multiple groups sliced from a long array to the cross-correlation calculation.",
            py::arg("starts"),
            py::arg("lengths"),
            py::arg("arr"),
            py::arg("autoConj") = true
        )
        .def("printGroups", &GroupXcorrCZT::printGroups)
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