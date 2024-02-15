#pragma once

#include "../../ipp_ext/include/ipp_ext.h"
#include <cmath>
#include <complex>

// =====================================
#ifdef COMPILE_FOR_PYBIND
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
#endif
// =====================================

// Helper functions
int next_fast_len(int len);

struct IppCZT32fc
{
    IppCZT32fc();
    IppCZT32fc(int len, double f1, double f2, double fstep, double fs);
    ~IppCZT32fc();

    IppCZT32fc(const IppCZT32fc& other) = delete;
    void operator=(const IppCZT32fc& other) = delete;

    //
    void prepare();
    void runRaw(const Ipp32fc* in, Ipp32fc* out);

    #ifdef COMPILE_FOR_PYBIND
    py::array_t<std::complex<float>, py::array::c_style> run(
        const py::array_t<std::complex<float>, py::array::c_style> &in
    );
    py::array_t<std::complex<float>, py::array::c_style> runMany(
        const py::array_t<std::complex<float>, py::array::c_style> &in
    );
    #endif


    //----------------------------------------------------------------
    // We are going to use list initialization, so the order of these member variables matters!
    int m_N;
    int m_k;
    int m_nfft;

    ippe::vector<Ipp32fc> m_ww;
    ippe::vector<Ipp32fc> m_fv;
    ippe::vector<Ipp32fc> m_aa;
    ippe::DFTCToC<Ipp32fc> m_dft;

    double m_f1;
    double m_f2;
    double m_fstep;
    double m_fs;

    ippe::vector<Ipp32fc> m_ws; // workspace to hold the input to the FFT
    ippe::vector<Ipp32fc> m_ws2; // workspace to hold the output from the FFT
};
