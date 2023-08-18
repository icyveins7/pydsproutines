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
    ippe::vector<Ipp32f> zeros(m_ww.size(), 0.0f);
    ippe::vector<Ipp32fc> ones32fc(m_ww.size());
    ippe::convert::RealToCplx()

    ippe::vector<Ipp32fc> chirpfilter(m_nfft);
    ippe::math::Div

    // nn = np.arange(self.m)
    // self.aa = np.exp(1j * 2 * np.pi * f1/fs * -nn) * self.ww[self.m+nn-1]
}

void IppCZT32fc::run(const Ipp32fc* in, Ipp32fc* out)
{

}