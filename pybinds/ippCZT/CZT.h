#pragma once

#include "../../ipp_ext/include/ipp_ext.h"
#include <cmath>

struct IppCZT32fc
{
    IppCZT32fc() {}
    IppCZT32fc(size_t len, Ipp32f f1, Ipp32f f2, Ipp32f fstep, Ipp32f fs)
        : m_len{len},
        m_k{static_cast<size_t>((f2-f1)/fstep + 1)},
        m_nfft{m_len + m_k}, // important for nfft to be after len and k
        m_dft{m_nfft}, // important for dft to be after nfft
        m_f1{f1},
        m_f2{f2},
        m_fstep{fstep},
        m_fs{fs}
    {
        prepare();
    }
    ~IppCZT32fc() {}

    //
    void prepare();
    void run(const Ipp32fc* in, Ipp32fc* out);


    //----------------------------------------------------------------
    // We are going to use list initialization, so the order of these member variables matters!
    size_t m_len;
    size_t m_k;
    size_t m_nfft;

    Ipp32f m_f1;
    Ipp32f m_f2;
    Ipp32f m_fstep;
    Ipp32f m_fs;
    ippe::vector<Ipp32fc> m_ww;
    ippe::vector<Ipp32fc> m_fv;
    ippe::vector<Ipp32fc> m_aa;
    ippe::DFTCToC<Ipp32fc> m_dft;
};
