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


//////////////// Constructors & Destructors /////////////////

IppCZT32fc::IppCZT32fc() 
{

}

IppCZT32fc::IppCZT32fc(int len, double f1, double f2, double fstep, double fs)
    : m_N{len},
    m_k{static_cast<int>((f2-f1)/fstep + 1)},
    m_nfft{next_fast_len(len + m_k - 1)}, // important for nfft to be after len and k
    m_dft{(size_t)m_nfft}, // important for dft to be after nfft
    m_f1{f1},
    m_f2{f2},
    m_fstep{fstep},
    m_fs{fs},
    m_ws{(size_t)m_nfft}, // again, important to be after m_nfft
    m_ws2{(size_t)m_nfft}
{
    // Validating the arguments
    // Frequency limits must be within +/- half of fs
    if (m_f1 <= -0.5*m_fs || m_f1 >= 0.5*m_fs)
        throw std::invalid_argument("f1 must be within +/- half of fs.");

    if (m_f2 <= -0.5*m_fs || m_f2 >= 0.5*m_fs)
        throw std::invalid_argument("f2 must be within +/- half of fs.");

    prepare();
}
IppCZT32fc::~IppCZT32fc()
{

}




// ----------------------------------------------------------------
void IppCZT32fc::prepare()
{
    // NOTE: we must compute all the required arrays in 64f/64fc and then convert down
    // otherwise the accuracy is very bad.. TODO: rewrite the below as 64f then convert at end
    try{
        // Start off by defining the array of k values, which spans [-N+1, max(N,K)-1], since we don't know if N or K is larger
        int kk_start = -m_N + 1;
        int kk_end = std::max(m_k, m_N) - 1; // inclusive
        ippe::vector<Ipp64f> kk(kk_end - kk_start + 1);
        ippe::generator::Slope<Ipp64f, Ipp64f>(
            kk.data(), 
            kk_end-kk_start+1, 
            (Ipp64f)kk_start, 
            1.0f);

        // Square the k values
        ippe::vector<Ipp64f> kk2(kk.size());
        ippe::math::Mul(
            kk.data(), kk.data(), kk2.data(), kk2.size()
        );
        // Then halve it
        ippe::math::MulC_I(0.5, kk2.data(), kk2.size());

        // Compute the W coefficients for the entire length, we will extract what we need later
        m_ww.resize(kk2.size()); // resize the actual, but we don't write to it
        ippe::vector<Ipp64fc> ww(m_ww.size()); // write to this temporary instead first

        Ipp64f f = -IPP_2PI * m_fstep / m_fs;
        ippe::math::MulC_I(f, kk2.data(), kk2.size());
        ippe::vector<Ipp64f> ones(kk2.size(), 1.0f);
        ippe::convert::PolarToCart(
            ones.data(), kk2.data(),
            ww.data(), ww.size()
        ); // ww is now filled completely and contains values from W^[-N+1, max(N,K)-1]^2/2
        // Convert to 32fc to save internally
        ippe::convert::Convert(
            reinterpret_cast<const Ipp64f*>(ww.data()), 
            reinterpret_cast<Ipp32f*>(m_ww.data()), \
            m_ww.size() * 2 // *2 for complex data
        );

        ippe::vector<Ipp64fc> ones64fc(m_ww.size());
        ippe::convert::RealToCplx(
            ones.data(), static_cast<Ipp64f*>(nullptr), ones64fc.data(), ones64fc.size()
        );

        Ipp64fc zero64fc = { 0.0, 0.0 };
        ippe::vector<Ipp64fc> chirpfilter(m_nfft, zero64fc);
        // Extract the negative powered chirpfilter V(n) by doing division, and extracting the first N+K-1, leaving the rest to be 0
        ippe::math::Div(ww.data(), ones64fc.data(), chirpfilter.data(), m_N + m_k - 1); // this is the ordering for 1/ww; ones32fc is the SECOND arg

        // Perform the FFT on the chirpfilter and store it
        m_fv.resize(m_nfft); // resize actual, but don't write to it
        ippe::vector<Ipp64fc> fv(m_nfft); // write to this temporary instead first
        ippe::DFTCToC<Ipp64fc> dft(m_nfft); // create a 64fc DFT object temporarily
        dft.fwd(chirpfilter.data(), fv.data()); // fv now contains the FFT of V(n)
        // Convert to 32fc to save internally
        ippe::convert::Convert(
            reinterpret_cast<const Ipp64f*>(fv.data()), 
            reinterpret_cast<Ipp32f*>(m_fv.data()), \
            m_fv.size() * 2 // *2 for complex data
        );

        // Now we need to calculate the coefficients to multiply with every new input array
        ippe::vector<Ipp64f> nn(m_N); // same length as input
        ippe::generator::Slope<Ipp64f, Ipp64f>(
            nn.data(), 
            nn.size(), 
            0.0, 
            1.0
        );
        // resizes
        m_aa.resize(nn.size()); // resize the actual output member var, but don't write to it
        ippe::vector<Ipp64fc> aa(m_aa.size()); // write to this temporary instead first
        ones.resize(nn.size(), 1.0); // resize ones to fit the correct length
        kk2.resize(nn.size(), 0.0); // resize kk2 and reuse it to store the phase for this
        // calculate the phase
        ippe::math::MulC(
            nn.data(), static_cast<Ipp64f>(-IPP_2PI * (m_f1 / m_fs)), 
            kk2.data(), kk2.size()
        );
        // exponentiate to complex
        ippe::convert::PolarToCart(
            ones.data(), kk2.data(), aa.data(), aa.size()
        );
        // final multiply
        ippe::math::Mul_I(
            &ww.at(m_N-1), // use the 64f temporary which is more accurate
            aa.data(), aa.size()
        );
        // Convert to 32fc to save internally
        ippe::convert::Convert(
            reinterpret_cast<const Ipp64f*>(aa.data()), 
            reinterpret_cast<Ipp32f*>(m_aa.data()), \
            m_aa.size() * 2 // *2 for complex data
        );
    }
    catch(std::exception &e){
        printf("Error caught in prepare(): %s\n", e.what());
        throw e;
    }
}

void IppCZT32fc::runRaw(const Ipp32fc* in, Ipp32fc* out)
{
    try{
        // Zero our workspace
        m_ws.zero();
        // Copy input to the front of workspace
        // printf("m_ws size: %zd\nin length: %d\n", m_ws.size(), m_N);
        ippe::Copy<Ipp32fc>(in, m_ws.data(), m_N); // only length N
        // Perform the multiply with our array of coefficients
        ippe::math::Mul_I(
            m_aa.data(), m_ws.data(), m_aa.size()
        );

        // Perform the FFT
        // printf("m_dft size: %zd\n", m_dft.getLength());
        // printf("m_ws2 size: %zd\n", m_ws2.size());
  //      printf("m_ws (%4zd): %p\n", m_ws.size(), m_ws.data());
  //      printf("m_ws2(%4zd): %p\n", m_ws.size(), m_ws2.data());
  //      printf("%p\n%p\n", m_dft.getDFTSpec().data(), m_dft.getDFTBuf().data());
  //      printf("%zd\n%zd\n", m_dft.getDFTSpec().size(), m_dft.getDFTBuf().size());
		//printf("%zd\n%zd\n", m_dft.getDFTSpec().capacity(), m_dft.getDFTBuf().capacity());
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
    catch(std::exception &e){
        printf("Error caught in runRaw(): %s\n", e.what());
        throw e;
    }
    
}

#ifdef COMPILE_FOR_PYBIND
py::array_t<std::complex<float>, py::array::c_style> IppCZT32fc::run(
    const py::array_t<std::complex<float>, py::array::c_style> &in
){
    // make sure it's 1D and correct length
    auto buffer_info = in.request();
    if (buffer_info.shape.size() != 1 || buffer_info.shape[0] != m_N)
        throw std::range_error("Input must be 1D and length " + std::to_string(m_N) + "!");

    // make the output
    py::array_t<std::complex<float>, py::array::c_style> out({m_k});
    
    try{
        const Ipp32fc* iptr = reinterpret_cast<const Ipp32fc*>(buffer_info.ptr);
        
        // Call the raw method
        runRaw(iptr,
            reinterpret_cast<Ipp32fc*>(out.request().ptr)  
        );
    }
    catch(...)
    {
        printf("Caught error?\n");
    }
    

    return out;
}

py::array_t<std::complex<float>, py::array::c_style> IppCZT32fc::runMany(
    const py::array_t<std::complex<float>, py::array::c_style> &in
){
    // make sure it's 2D and correct columns
    auto buffer_info = in.request();
    if (buffer_info.shape.size() != 2 || buffer_info.shape[1] != m_N)
        throw std::range_error("Input must be 2D and columns = " + std::to_string(m_N) + "!");

    // make the output
    py::array_t<std::complex<float>, py::array::c_style> out({(int)buffer_info.shape[0], m_k});
    

    try{
        const Ipp32fc* iptr = reinterpret_cast<const Ipp32fc*>(buffer_info.ptr);
        Ipp32fc* optr = reinterpret_cast<Ipp32fc*>(out.request().ptr);

        // Call the raw method over every row
        for (int i = 0; i < buffer_info.shape[0]; i++)
        {
            runRaw(&iptr[i * m_N],
                reinterpret_cast<Ipp32fc*>(&optr[i*m_k])  
            );
        }
        
    }
    catch(...)
    {
        printf("Caught error?\n");
    }
    

    return out;
}
#endif