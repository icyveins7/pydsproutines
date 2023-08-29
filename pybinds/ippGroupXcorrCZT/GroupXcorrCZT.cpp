#include "GroupXcorrCZT.h"

void GroupXcorrCZT::addGroup(int start, int length, Ipp32fc *group, bool autoConj)
{
    // Add the indices
    if (length > m_czt.m_N)
        throw std::range_error("Length of group exceeds maximum length");
    m_groupStarts.push_back(start);
    m_groupLengths.push_back(length);

    // Copy the group itself
    ippe::vector<Ipp32fc> groupVector(length);
    // Conjugate if needed
    if (autoConj)
        ippe::convert::Conj(group, groupVector.data(), length);
    else
        ippe::Copy(group, groupVector.data(), length);
    m_groups.push_back(std::move(groupVector)); // TODO: check if move works

    // Calculate the energy of the group
    m_groupEnergies.push_back(0.0);
    ippe::stats::Norm_L2(group, length, &m_groupEnergies.back());
    m_groupEnergies.back() = m_groupEnergies.back() * m_groupEnergies.back(); // remember to square to get energy
}

void GroupXcorrCZT::resetGroups()
{
    m_groupStarts.clear();
    m_groupLengths.clear();
    m_groups.clear();
    m_groupEnergies.clear();
}

void GroupXcorrCZT::xcorr(
    Ipp32fc *x, int shiftStart, int shiftStep, int numShifts, Ipp32fc *output
){
    int shift;
    ippe::vector<Ipp32fc> pdt(m_czt.m_N);
    ippe::vector<Ipp32fc> result(m_czt.m_k);
    m_xEnergies.resize(numShifts, 0.0);
    // Output is assumed to be of size numShifts * m_czt.m_k
    // It is also assumed to be zeroed already!

    // Calculate the total energy of all the groups
    Ipp64f totalGroupEnergy;
    // TODO: ippe sum over groupEnergies

    // Loop over the group
    for (int i = 0; i < m_groupStarts.size(); i++)
    {
        // Loop over the shifts
        for (int j = 0; j < numShifts; j++)
        {
            shift = shiftStart + j * shiftStep;

            // Calculate the energy for this slice of x
            Ipp64f energy;
            ippe::stats::Norm_L2(
                &x[shift],
                m_groupLengths.at(i),
                &energy
            );
            // Accumulate energy for this shift
            m_xEnergies.at(j) += energy*energy; // remember to square to get energy

            // Multiply by the group
            pdt.zero(); // we must zero since the pdt may be longer than the current group length
            ippe::math::Mul(
                m_groups.at(i).data(),
                &x[shift],
                pdt.data(),
                m_groupLengths.at(i)
            );

            // Run the CZT
            m_czt.runRaw(pdt.data(), result.data()); // TODO: make CZT not copy this

            // Multiply in the correction for this group
            ippe::math::Mul_I(
                m_groupPhaseCorrections.at(i).data(),
                result.data(),
                (int)result.size()
            );

            // Accumulate into the output
            ippe::math::Add_I(
                result.data(),
                &output[i * m_czt.m_k],
                (int)result.size()
            );
        }
    }

    // After the 2 loops, we perform normalisation of the output
    for (int j = 0; j < numShifts; j++)
    {
        Ipp64f normalisation = m_xEnergies.at(j) * totalGroupEnergy;
        Ipp32fc normalisation_c = {
            static_cast<Ipp32f>(1.0/normalisation),
            0.0f
        };
        // Normalise by multiplying 1 / normalisation, using a complex value
        ippe::math::MulC_I(
            normalisation_c,
            &output[j * m_czt.m_k],
            (int)m_czt.m_k
        );
    }
}

////////////////////////////////////////////////////////////////////////
void GroupXcorrCZT::computeGroupPhaseCorrections()
{
    m_groupPhaseCorrections.clear();
    m_groupPhaseCorrections.resize(m_groupStarts.size());

    // Phase correction for each frequency of CZT
    ippe::vector<Ipp64f> phase(m_czt.m_k);
    ippe::vector<Ipp64fc> correction(phase.size()); // temporary 64-bit vector
    ippe::vector<Ipp64f> ones(phase.size(), 1.0);
    for (int i = 0; i < m_groupStarts.size(); i++)
    {
        // Set the frequency slope of the CZT in the phase
        ippe::generator::Slope(
            phase.data(),
            (int)phase.size(),
            m_czt.m_f1 / m_czt.m_fs,
            m_czt.m_fstep
        );
        // Compute -2pi * f * groupStart
        ippe::math::MulC_I(
            -IPP_2PI * (Ipp64f)m_groupStarts.at(i),
            phase.data(),
            (int)phase.size()
        );
        // Exponentiate to get the complex vector
        ippe::convert::PolarToCart(
            ones.data(),
            phase.data(),
            correction.data(),
            (int)correction.size()
        );
        // Convert back down to 32fc and store
        m_groupPhaseCorrections.at(i).resize(phase.size());
        ippe::convert::Convert(
            correction.data(),
            m_groupPhaseCorrections.at(i).data(),
            (int)m_groupPhaseCorrections.at(i).size()
        );
    }
}