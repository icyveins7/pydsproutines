#include <iostream>
#include "ipp_ext.h"

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
        // DEBUG checking
        for (auto p : m_phaseOffset)
        {
            printf("%f rad\n", p);
        }
        // Add the original phases into phaseOffset vector
        ippe::math::Add_I(
            m_phase.data(),
            m_phaseOffset.data(),
            (int)m_phase.size()
        );
        // Convert back to cartesian
        ippe::convert::PolarToCartDeinterleaved(
            m_magn.data(),
            m_phaseOffset.data(),
            re,
            im,
            (int)m_magn.size()
        );
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
    ) : m_x{len}
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
        // TODO


    }

private:
    ippe::vector<Ipp32fc> m_x;
    ippe::vector<Ipp32fc> m_tone;

};

int main()
{
    // Make some data
    ippe::vector<Ipp32fc> syms(4);
    syms.at(0) = {1.0f, 0.0f};
    syms.at(1) = {0.0f, 1.0f};
    syms.at(2) = {-1.0f, 0.0f};
    syms.at(3) = {0.0f, -1.0f};

    try
    {
        FrequencyAdjusterMethod1 adj(syms.data(), (int)syms.size());

        ippe::vector<Ipp32f> re(syms.size());
        ippe::vector<Ipp32f> im(syms.size());

        adj.adjust(0.125f, 0.0f, re.data(), im.data()); 

        for (int i = 0; i < (int)re.size(); i++)
        {
            printf("%f, %f\n", re.at(i), im.at(i));
        }
        printf("=========\n");


    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
    

    return 0;
}