#include <iostream>
#include "ipp_ext.h"
#include <chrono>

class HighResolutionTimer
{
public:
    HighResolutionTimer()
    {
        t1 = std::chrono::high_resolution_clock::now();
    }

    ~HighResolutionTimer()
    {
        t2 = std::chrono::high_resolution_clock::now();
        printf("%fs elapsed.\n", std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count());
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> t1;
    std::chrono::time_point<std::chrono::high_resolution_clock> t2;
};


class FrequencyAdjusterMethod1
{
public:
    FrequencyAdjusterMethod1(const Ipp32fc* x, const size_t len)
        : m_magn{len}, m_phase{len}, m_phaseOffset{len}
    {
        ippe::convert::CartToPolar(
            x, m_magn.data(), m_phase.data(), len
        );
    }

    /// @brief 
    /// @param freq Normalised frequency i.e. f/fs
    /// @param phase0 
    /// @param re 
    /// @param im 
    /// @param size 
    void adjust(float freq, float phase0, float *re, float *im)
    {
        // Calculate phase offsets
        ippe::generator::Slope(
            m_phaseOffset.data(),
            (int)m_phaseOffset.size(),
            phase0,
            static_cast<float>(freq*IPP_2PI)
        );
        // // DEBUG checking
        // for (auto p : m_phaseOffset)
        // {
        //     printf("%f rad\n", p);
        // }
        // Add the original phases into phaseOffset vector
        ippe::math::Add_I(
            m_phase.data(),
            m_phaseOffset.data(),
            (int)m_phase.size()
        );
        // // Convert back to cartesian
        // ippe::convert::PolarToCartDeinterleaved(
        //     m_magn.data(),
        //     m_phaseOffset.data(),
        //     re,
        //     im,
        //     (int)m_magn.size()
        // );
    };

private:
    ippe::vector<Ipp32f> m_magn;
    ippe::vector<Ipp32f> m_phase;
    ippe::vector<Ipp32f> m_phaseOffset;
};

class FrequencyAdjusterMethod2
{
public:
    FrequencyAdjusterMethod2(
        const Ipp32fc *x, const size_t len
    ) : m_x{len}, m_tone{len}
    {
        ippe::Copy(x, m_x.data(), (int)len);
    }

    void adjust(float freq, float phase0, float *re, float *im)
    {
        // Generate tone
        ippe::generator::Tone(
            m_tone.data(),
            (int)m_tone.size(),
            1.0f,
            freq,
            &phase0
        );
        
        // Multiply tone
        ippe::math::Mul_I(
            m_x.data(),
            m_tone.data(), // stored inside tone now
            (int)m_x.size()
        );

        // // Split into real and imaginary parts
        // ippe::convert::CplxToReal(
        //     m_tone.data(),
        //     re,
        //     im,
        //     (int)m_tone.size()
        // );
    }

private:
    ippe::vector<Ipp32fc> m_x;
    ippe::vector<Ipp32fc> m_tone;

};

int main()
{
    size_t len = 1000000;    

    // Make some data
    ippe::vector<Ipp32fc> syms(len);
    syms.at(0) = {1.0f, 0.0f};
    syms.at(1) = {0.0f, 1.0f};
    syms.at(2) = {-1.0f, 0.0f};
    syms.at(3) = {0.0f, -1.0f};

    try
    {
        FrequencyAdjusterMethod1 adj(syms.data(), (int)syms.size());
        FrequencyAdjusterMethod2 adj2(syms.data(), (int)syms.size());

        ippe::vector<Ipp32f> re(syms.size());
        ippe::vector<Ipp32f> im(syms.size());


        // Method 1
        printf("Method1 \n");
        {
            HighResolutionTimer t;
            adj.adjust(0.125f, 0.0f, re.data(), im.data()); 
        }

        
        // for (int i = 0; i < (int)re.size(); i++)
        // {
        //     printf("%f, %f\n", re.at(i), im.at(i));
        // }
        printf("=========\n");

        // Method 2
        printf("Method2\n");
        {
            HighResolutionTimer t;
            adj2.adjust(0.125f, 0.0f, re.data(), im.data());
        }

        

        // for (int i = 0; i < (int)re.size(); i++)
        // {
        //     printf("%f, %f\n", re.at(i), im.at(i));
        // }
        printf("=========\n");


    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
    

    return 0;
}