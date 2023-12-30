#include <iostream>
#include "../../ipp_ext/include/ipp_ext.h"

class FrequencyAdjuster
{
public:
    void adjust(float freq, float *re, float *im, int size)
    {
        throw std::runtime_error(
            "Frequency adjuster is not implemented yet."
        );
    }
};

class FrequencyAdjusterMethod1 : public FrequencyAdjuster
{
public:
    FrequencyAdjusterMethod1(Ipp32f* x, int len)
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
        m_phaseOffset.zero();
        ippe::generator::Slope(
            m_phaseOffset.data(),
            (int)m_phaseOffset.size(),
            phase0,
            freq*IPP_2PI
        );
        // Add the original phases
        ippe::math::Add_I(
            m_phase.data(),
            m_phaseOffset.data(),
            (int)m_phase.size()
        );
        // Convert back to cartesian
        
    }

private:
    ippe::vector<Ipp32f> m_magn;
    ippe::vector<Ipp32f> m_phase;
    ippe::vector<Ipp32f> m_phaseOffset;
};