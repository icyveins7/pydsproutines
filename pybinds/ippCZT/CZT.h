#pragma once

#include "../../ipp_ext/include/ipp_ext.h"
#include <cmath>
#include <complex>

// =====================================
// #include <pybind11/numpy.h>
// #include <pybind11/complex.h>
// #include <pybind11/pybind11.h>

// namespace py = pybind11;
// =====================================

// Helper functions
int next_fast_len(int len);

struct IppCZT32fc
{
    IppCZT32fc() {}
    IppCZT32fc(int len, Ipp32f f1, Ipp32f f2, Ipp32f fstep, Ipp32f fs)
        : m_N{len},
        m_k{static_cast<int>((f2-f1)/fstep + 1)},
        m_nfft{next_fast_len(len + m_k - 1)}, // important for nfft to be after len and k
        m_dft{m_nfft}, // important for dft to be after nfft
        m_f1{f1},
        m_f2{f2},
        m_fstep{fstep},
        m_fs{fs},
        m_ws{m_nfft}, // again, important to be after m_nfft
        m_ws2{m_nfft}
    {
        prepare();
    }
    ~IppCZT32fc() {}

    //
    void prepare();
    void runRaw(const Ipp32fc* in, Ipp32fc* out);
    // py::array_t<std::complex<float>, py::array::c_style> run(
    //     const py::array_t<std::complex<float>, py::array::c_style> &in
    // );


    //----------------------------------------------------------------
    // We are going to use list initialization, so the order of these member variables matters!
    int m_N;
    int m_k;
    int m_nfft;

    Ipp32f m_f1;
    Ipp32f m_f2;
    Ipp32f m_fstep;
    Ipp32f m_fs;
    ippe::vector<Ipp32fc> m_ww;
    ippe::vector<Ipp32fc> m_fv;
    ippe::vector<Ipp32fc> m_aa;
    ippe::DFTCToC<Ipp32fc> m_dft;

    ippe::vector<Ipp32fc> m_ws; // workspace to hold the input to the FFT
    ippe::vector<Ipp32fc> m_ws2; // workspace to hold the output from the FFT
};
