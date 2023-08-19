#include "CZT.h"

void IppCZT32fc::prepare()
{
    // kk = np.arange(-self.m+1,np.max([self.k-1,self.m-1])+1)
    // kk2 = kk**2.0 / 2.0
    int kk_start = -m_len + 1;
    int kk_end = std::max(m_k, m_len);
    ippe::vector<Ipp32f> kk(kk_end - kk_start + 1);
    ippe::generator::Slope<Ipp32f, Ipp32f>(
        kk.data(), 
        kk_end-kk_start+1, 
        (Ipp32f)kk_start, 
        1.0f);

    ippe::vector<Ipp32f> kk2(kk.size());
    ippe::math::Mul(
        kk.data(), kk.data(), kk2.data(), kk2.size()
    );
    ippe::math::MulC_I(0.5f, kk2.data(), kk2.size());

    // self.ww = np.exp(-1j * 2 * np.pi * (f2-f1+binWidth)/(self.k*fs) * kk2)
    // resize m_ww appropriately
    m_ww.resize(kk2.size());
    Ipp32f f = -(m_f2-m_f1+m_fstep) / ((Ipp32f)m_k * m_fs) * 6.283185307179586;
    ippe::math::MulC_I(f, kk2.data(), kk2.size());
    ippe::vector<Ipp32f> ones(kk2.size(), 1.0f);
    ippe::convert::PolarToCart(
        ones.data(), kk2.data(),
        m_ww.data(), m_ww.size()
    );

    // chirpfilter = 1 / self.ww[:self.k-1+self.m]
    // self.fv = np.fft.fft( chirpfilter, self.nfft )
    ippe::vector<Ipp32fc> ones32fc(m_ww.size());
    ippe::convert::RealToCplx(
        ones.data(), static_cast<Ipp32f*>(nullptr), ones32fc.data(), ones32fc.size()
    );

    ippe::vector<Ipp32fc> chirpfilter(m_nfft);
    ippe::math::Div(ones32fc.data(), m_ww.data(), chirpfilter.data(), m_nfft);
    // TODO: Warp nfft to the next best length

    // TODO: perform the FFT


    // nn = np.arange(self.m)
    // self.aa = np.exp(1j * 2 * np.pi * f1/fs * -nn) * self.ww[self.m+nn-1]
    ippe::vector<Ipp32f> nn(m_len);
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
        &m_ww.at(m_len-1), m_aa.data(), m_aa.size()
    );
}

void IppCZT32fc::run(const Ipp32fc* in, Ipp32fc* out)
{

}