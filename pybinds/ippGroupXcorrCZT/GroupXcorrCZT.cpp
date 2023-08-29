#include "GroupXcorrCZT.h"

void GroupXcorrCZT::addGroup(int start, int length, Ipp32fc *group, bool autoConj)
{
    // Before we add the group, check that it doesn't overlap with any existing groups
    for (int i = 0; i < m_groups.size(); i++)
    {
        int gStart = m_groupStarts.at(i);
        int gEnd = gStart + (int)m_groups.at(i).size();
        if (start >= gStart && start < gEnd)
            throw std::range_error(
                "Group start overlaps with existing group! [" + std::to_string(gStart) + "," + 
                std::to_string(gEnd) + ")"
            );
        if (start + length >= gStart && start + length < gEnd)
            throw std::range_error(
                "Group end overlaps with existing group! [" + std::to_string(gStart) + "," + 
                std::to_string(gEnd) + ")"
            );
    }
    // Make sure the length doesn't exceed the max length
    if (length > m_czt.m_N)
        throw std::range_error("Length of group exceeds maximum length");
        
    // If okay, then add the indices
    m_groupStarts.push_back(start);

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
    // m_groupLengths.clear();
    m_groups.clear();
    m_groupEnergies.clear();
}

void GroupXcorrCZT::xcorr(
    Ipp32fc *x, int shiftStart, int shiftStep, int numShifts, Ipp32f *out
){
    int shift;
    ippe::vector<Ipp32fc> pdt(m_czt.m_N);
    ippe::vector<Ipp32fc> result(m_czt.m_k);
    // m_xEnergies.resize(numShifts, 0.0);
    ippe::vector<Ipp64f> xEnergies(numShifts, 0.0);
    ippe::vector<Ipp32fc> accumulator(numShifts * m_czt.m_k, {0.0f, 0.0f});
    // Output is assumed to be of size numShifts * m_czt.m_k
    // It is also assumed to be zeroed already!

    // Calculate the total energy of all the groups
    Ipp64f totalGroupEnergy;
    ippe::stats::Sum(
        m_groupEnergies.data(), (int)m_groupEnergies.size(), &totalGroupEnergy
    );
    printf("totalGroupEnergy = %.8f\n", totalGroupEnergy);

    // Compute all the group phase corrections
    computeGroupPhaseCorrections();
    printf("Completed computing group phase corrections\n");
    for (int i = 0; i < m_groupPhaseCorrections.size(); ++i)
    {
        printf("Group %d corrections\n", i);
        for (int j = 0; j < m_groupPhaseCorrections[i].size(); ++j)
        {
            printf("%g %g\n",
                m_groupPhaseCorrections[i][j].re, m_groupPhaseCorrections[i][j].im
            );
        }
    }

    // Loop over the groups
    for (int i = 0; i < m_groupStarts.size(); i++)
    {
        // Loop over the shifts
        for (int j = 0; j < numShifts; j++)
        {
            shift = shiftStart + j * shiftStep;
            printf("Group %d(length %zd), shift[%d] = %d\n", 
                i, m_groups.at(i).size(), j, shift);
            int xi = shift + m_groupStarts.at(i);
            printf("xi = %d\n", xi);
            
            // Calculate the energy for this slice of x
            Ipp64f energy;
            ippe::stats::Norm_L2(
                &x[xi],
                (int)m_groups.at(i).size(),
                &energy
            );
            printf("Energy for this slice of x = %f\n", energy);
            // Accumulate energy for this shift
            xEnergies.at(j) += energy*energy; // remember to square to get energy
            printf("xEnergies[%d] = %f\n", j, xEnergies.at(j));

            // Multiply by the group
            pdt.zero(); // we must zero since the pdt may be longer than the current group length
            ippe::math::Mul(
                m_groups.at(i).data(),
                &x[xi], 
                pdt.data(), // output to the beginning of the buffer
                (int)m_groups.at(i).size() // only multiply up to the current group length
            );

            // Run the CZT
            m_czt.runRaw(pdt.data(), result.data()); // TODO: make CZT not copy this

            // Multiply in the correction for this group
            ippe::math::Mul_I(
                m_groupPhaseCorrections.at(i).data(),
                result.data(),
                (int)result.size()
            );
            // DEBUG
            for (int j = 0; j < m_czt.m_k; j++){
                printf("%.6g,%.6g \n", result[j].re, accumulator[j].im);
            }

            // Accumulate into the output
            ippe::math::Add_I(
                result.data(),
                &accumulator[j * m_czt.m_k],
                (int)result.size()
            );
        }
    }

    // DEBUG
    for (int i = 0; i < numShifts; i++){
        for (int j = 0; j < m_czt.m_k; j++){
            printf("%.6g,%.6g  ", accumulator[i*m_czt.m_k + j].re, accumulator[i*m_czt.m_k + j].im);
        }
        printf("\n");
    }

    // Take the abs sq of the output
    ippe::convert::PowerSpectr(
        accumulator.data(),
        out,
        (int)accumulator.size()
    );

    // DEBUG
    for (int i = 0; i < numShifts; i++){
        for (int j = 0; j < m_czt.m_k; j++){
            printf("%.6g  ", out[i*m_czt.m_k + j]);
        }
        printf("\n");
    }

    

    // After the 2 loops, we perform normalisation of the output
    for (int j = 0; j < numShifts; j++)
    {
        Ipp64f normalisation = 1.0/(xEnergies.at(j) * totalGroupEnergy);
        printf("Normalisation for shift %d is %g\n", j, normalisation);

        // Ipp32fc normalisation_c = {
        //     static_cast<Ipp32f>(normalisation),
        //     0.0f
        // };
        // printf("Normalisation for shift %d is %g,%g\n", j,
        //     normalisation_c.re, normalisation_c.im
        // );

        // Normalise by multiplying 1 / normalisation, using a complex value
        ippe::math::MulC_I(
            static_cast<Ipp32f>(normalisation),
            &out[j * m_czt.m_k],
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
        ippe::convert::Convert( // remember we have to reinterpret as real arrays
            reinterpret_cast<const Ipp64f*>(correction.data()),
            reinterpret_cast<Ipp32f*>(m_groupPhaseCorrections.at(i).data()),
            (int)m_groupPhaseCorrections.at(i).size() * 2
        );
    }
}