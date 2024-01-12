#include "ffs/include/ffs.h"

#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

// Define the pybind wrapper(s)
template <typename T, size_t UNROLL = 1>
void freqshift(
    py::array_t<std::complex<T>, py::array::c_style> &in,
    T freq, T phase
){
    // make sure it's 1D
    auto buffer_info = in.request();
    if (buffer_info.shape.size() != 1)
        throw std::range_error("Input must be 1D");
    if (freq < 0 || freq >= 1)
        throw std::range_error("Frequency must be in [0, 1)");
    
    try{
        // Call ffs
        ffs::shiftArray<T, UNROLL>(
            reinterpret_cast<std::complex<T>*>(buffer_info.ptr), 
            buffer_info.shape[0],
            freq, phase);
    }
    catch(...)
    {
        printf("Caught error?\n");
    }
}

// This module name in PYBIND11_MODULE must be the same as the filename!
// https://github.com/pybind/pybind11/blob/master/docs/faq.rst
PYBIND11_MODULE(pbffs, m) {
    m.def("ffs1_32fc", &freqshift<float,1>,
        "Fast frequency shift in-place for 32-bit floats, optimization UNROLL 1",
        py::arg("in"), py::arg("freq"), py::arg("phase")
    );
    m.def("ffs4_32fc", &freqshift<float,4>,
        "Fast frequency shift in-place for 32-bit floats, optimization UNROLL 4",
        py::arg("in"), py::arg("freq"), py::arg("phase")
    );
    m.def("ffs1_64fc", &freqshift<double,1>,
        "Fast frequency shift in-place for 64-bit floats, optimization UNROLL 1",
        py::arg("in"), py::arg("freq"), py::arg("phase")
    );
    m.def("ffs4_64fc", &freqshift<double,4>,
        "Fast frequency shift in-place for 64-bit floats, optimization UNROLL 4",
        py::arg("in"), py::arg("freq"), py::arg("phase")
    );

    m.doc() = "pybind11 for ffs"; // optional module docstring
}