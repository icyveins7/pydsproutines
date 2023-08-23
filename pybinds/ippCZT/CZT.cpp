#include "CZT.h"

int next_fast_len(int len)
{
    // We check 2,3,5,7
    int primes[4] = { 2, 3, 5, 7 };

    bool fastlenFound = false;
    int addition = 0;
    while (!fastlenFound)
    {
        int n = len + addition;
        for (int i = 0; i < 4; i++)
        {
            while (n % primes[i] == 0)
            {
                n /= primes[i];
            }
        }

        if (n == 1)
        {
            fastlenFound = true;
            break;
        }
        else
            addition++;
    }

    return len+addition;
}

// ----------------------------------------------------------------
void IppCZT32fc::prepare()
{
    // Start off by defining the array of k values, which spans [-N+1, max(N,K)-1], since we don't know if N or K is larger
    int kk_start = -m_N + 1;
    int kk_end = std::max(m_k, m_N) - 1; // inclusive
    ippe::vector<Ipp32f> kk(kk_end - kk_start + 1);
    ippe::generator::Slope<Ipp32f, Ipp32f>(
        kk.data(), 
        kk_end-kk_start+1, 
        (Ipp32f)kk_start, 
        1.0f);

    // Square the k values
    ippe::vector<Ipp32f> kk2(kk.size());
    ippe::math::Mul(
        kk.data(), kk.data(), kk2.data(), kk2.size()
    );
    // Then halve it
    ippe::math::MulC_I(0.5f, kk2.data(), kk2.size());

    // Compute the W coefficients for the entire length, we will extract what we need later
    m_ww.resize(kk2.size());
    Ipp32f f = -6.283185307179586 * m_fstep / m_fs;
    // Ipp32f f = -(m_f2-m_f1+m_fstep) / ((Ipp32f)m_k * m_fs) * 6.283185307179586;
    ippe::math::MulC_I(f, kk2.data(), kk2.size());
    ippe::vector<Ipp32f> ones(kk2.size(), 1.0f);
    ippe::convert::PolarToCart(
        ones.data(), kk2.data(),
        m_ww.data(), m_ww.size()
    ); // m_ww is now filled completely and contains values from W^[-N+1, max(N,K)-1]^2/2

    ippe::vector<Ipp32fc> ones32fc(m_ww.size());
    ippe::convert::RealToCplx(
        ones.data(), static_cast<Ipp32f*>(nullptr), ones32fc.data(), ones32fc.size()
    );

    Ipp32fc zero32fc = { 0.0f, 0.0f };
    ippe::vector<Ipp32fc> chirpfilter(m_nfft, zero32fc);
    // Extract the negative powered chirpfilter V(n) by doing division, and extracting the first N+K-1, leaving the rest to be 0
    ippe::math::Div(ones32fc.data(), m_ww.data(), chirpfilter.data(), m_N + m_k - 1);
    for (int i = 0; i < chirpfilter.size(); i++)
        printf("chirpfilter[%d] = %f %f\n", i, chirpfilter[i].re, chirpfilter[i].im);

    // Perform the FFT on the chirpfilter and store it
    m_fv.resize(m_nfft);
    m_dft.fwd(chirpfilter.data(), m_fv.data()); // m_fv now contains the FFT of V(n)

    // Now we need to calculate the coefficients to multiply with every new input array
    ippe::vector<Ipp32f> nn(m_N); // same length as input
    ippe::generator::Slope<Ipp32f, Ipp32f>(
        nn.data(), 
        nn.size(), 
        0.0f, 
        1.0f
    );
    // resizes
    m_aa.resize(nn.size()); // resize the actual output member var
    ones.resize(nn.size(), 1.0f); // resize ones to fit the correct length
    kk2.resize(nn.size(), 0.0f); // resize kk2 and reuse it to store the phase for this
    // calculate the phase
    ippe::math::MulC(
        nn.data(), static_cast<Ipp32f>(-6.283185307179586 * (m_f1 / m_fs)), 
        kk2.data(), kk2.size()
    );
    // exponentiate to complex
    ippe::convert::PolarToCart(
        ones.data(), kk2.data(), m_aa.data(), m_aa.size()
    );
    // final multiply
    ippe::math::Mul_I(
        &m_ww.at(m_N-1), m_aa.data(), m_aa.size()
    );
}

void IppCZT32fc::run(const Ipp32fc* in, Ipp32fc* out)
{
    // Zero our workspace
    m_ws.zero();
    // Copy input to the front of workspace
    ippe::Copy<Ipp32fc>(in, m_ws.data(), m_N); // only length N
    // Perform the multiply with our array of coefficients
    ippe::math::Mul_I(
        m_aa.data(), m_ws.data(), m_ws.size()
    );
    // Perform the FFT
    m_dft.fwd(m_ws.data(), m_ws2.data());
    // Multiply with the chirpfilter FFT
    ippe::math::Mul_I(
        m_fv.data(), m_ws2.data(), m_ws.size()
    );
    // Perform the IFFT
    m_dft.bwd(m_ws2.data(), m_ws.data()); // reuse m_ws to hold the output again
    // Multiply the correct section with the post coefficients slice appropriately
    ippe::math::Mul(
        &m_ws.at(m_N-1), &m_ww.at(m_N-1), out, m_k
    );

}