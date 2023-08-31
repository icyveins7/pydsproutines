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
    if (length > m_czts.at(0).m_N)
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
    m_groups.push_back(std::move(groupVector)); // the move works! note that the push_back invokes an internal resize, which still performs a copy on all current elements..

    // Calculate the energy of the group
    m_groupEnergies.push_back(0.0);
    ippe::stats::Norm_L2(group, length, &m_groupEnergies.back());
    m_groupEnergies.back() = m_groupEnergies.back() * m_groupEnergies.back(); // remember to square to get energy
}


void GroupXcorrCZT::resetGroups()
{
    m_groupStarts.clear();
    m_groups.clear();
    m_groupEnergies.clear();
}

void GroupXcorrCZT::xcorrRaw(
    Ipp32fc *x, int shiftStart, int shiftStep, int numShifts, Ipp32f *out
){
    // Check if any groups have been defined
    if (m_groupStarts.size() == 0)
        throw std::range_error("No groups have been defined!");

    // Calculate the total energy of all the groups
    Ipp64f totalGroupEnergy; // this can be done in main thread as it's pretty cheap
    ippe::stats::Sum(
        m_groupEnergies.data(), (int)m_groupEnergies.size(), &totalGroupEnergy
    );
    // printf("totalGroupEnergy = %.8f\n", totalGroupEnergy);

    // Compute all the group phase corrections
    m_groupPhaseCorrections.clear();
    m_groupPhaseCorrections.resize(m_groupStarts.size());
    if (m_threads.size() > 1){
        for (int t = 0; t < m_threads.size(); t++){
            m_threads[t] = std::thread(
                &GroupXcorrCZT::computeGroupPhaseCorrections,
                this,
                t,
                (int)m_threads.size()
            );
        }
        for (int t = 0; t < m_threads.size(); t++){
            m_threads[t].join();
        }       
    }
    else // single-threaded
        computeGroupPhaseCorrections();

    // // DEBUG phase corrections
    // for (int i = 0; i < m_groupPhaseCorrections.size(); i++){
    //     for (int j = 0; j < m_groupPhaseCorrections.at(i).size(); j++){
    //         printf("%.8f %.8fi\n", m_groupPhaseCorrections.at(i).at(j).re, m_groupPhaseCorrections.at(i).at(j).im);
    //     }
    //     printf("\n");
    // }

    // invoke the main loops
    if (m_threads.size() > 1)
    {
        for (int t = 0; t < m_threads.size(); t++){
            m_threads[t] = std::thread(
                &GroupXcorrCZT::correlateGroups,
                this,
                x,
                shiftStart,
                shiftStep,
                numShifts,
                totalGroupEnergy,
                out,
                t,
                (int)m_threads.size()
            );
        }
        for (int t = 0; t < m_threads.size(); t++){
            m_threads[t].join();
        }
    }
    else // single-threaded
        correlateGroups(
            x,
            shiftStart,
            shiftStep,
            numShifts,
            totalGroupEnergy,
            out
        );


}

////////////////////////////////////////////////////////////////////////
void GroupXcorrCZT::correlateGroups(
    Ipp32fc *x, 
    int shiftStart, int shiftStep, int numShifts, 
    Ipp64f totalGroupEnergy,
    Ipp32f *out,
    int t, int NUM_THREADS
){
    // Retrieve the CZT object used by this thread
    IppCZT32fc &m_czt = m_czts.at(t);

    // Define the shifts that this thread works on
    int NUM_PER_THREAD = numShifts / NUM_THREADS;
    NUM_PER_THREAD = numShifts % NUM_THREADS > 0 ? NUM_PER_THREAD + 1 : NUM_PER_THREAD; // increment if leftovers
    int t_jStart = t * NUM_PER_THREAD; // first shift index for the thread
    int t_jEnd = (t + 1) * NUM_PER_THREAD > numShifts ? numShifts : (t + 1) * NUM_PER_THREAD; // last shift index for the thread
    // printf("Thread %d: %d to %d\n", t, t_jStart, t_jEnd);
    if (t_jStart - t_jEnd == 0) // then this thread has nothing to do
        return;

    // Allocate workspace for the thread
    int shift;
    ippe::vector<Ipp32fc> pdt(m_czt.m_N);
    ippe::vector<Ipp32fc> result(m_czt.m_k);
    
    // allocate exactly enough for these 2
    ippe::vector<Ipp32fc> accumulator((t_jEnd - t_jStart) * m_czt.m_k, {0.0f, 0.0f});
    ippe::vector<Ipp64f> xEnergies(t_jEnd - t_jStart, 0.0);
    // Output is assumed to be of size numShifts * m_czt.m_k
    // It is also assumed to be zeroed already!

    // Loop over the groups
    for (int i = 0; i < m_groupStarts.size(); i++)
    {
        // Loop over the shifts 
        // here is where we split the work over threads
        // as this allows us to avoid race conditions on the accumulator
        // i.e. no more than 1 thread writes to 1 shift
        for (int j = 0; j < t_jEnd - t_jStart; j++)
        {
            // offset appropriately for this thread
            shift = shiftStart + (j + t_jStart) * shiftStep;
            // printf("Group %d(length %zd), shift[%d] = %d\n", 
            //     i, m_groups.at(i).size(), j, shift);
            int xi = shift + m_groupStarts.at(i);
            // printf("xi = %d\n", xi);
            
            // Calculate the energy for this slice of x
            Ipp64f energy;
            ippe::stats::Norm_L2(
                &x[xi],
                (int)m_groups.at(i).size(),
                &energy
            );
            // printf("Energy for this slice of x = %f\n", energy);
            // Accumulate energy for this shift
            xEnergies.at(j) += energy*energy; // remember to square to get energy
            // printf("xEnergies[%d] = %f\n", j, xEnergies.at(j));

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

            // Accumulate into the output
            ippe::math::Add_I(
                result.data(),
                &accumulator[j * m_czt.m_k],
                (int)result.size()
            );
        }
    }

    // // DEBUG
    // for (int i = 0; i < numShifts; i++){
    //     for (int j = 0; j < m_czt.m_k; j++){
    //         printf("%.6g,%.6g  ", accumulator[i*m_czt.m_k + j].re, accumulator[i*m_czt.m_k + j].im);
    //     }
    //     printf("\n");
    // }

    // Take the abs sq of the output, but only the section this thread works on
    ippe::convert::PowerSpectr(
        accumulator.data(), // read from our threaded workspace
        &out[t_jStart * m_czt.m_k], // to where it's supposed to be written in the full output
        (int)accumulator.size() // the threaded workspace has the exact size needed
    );

    // // DEBUG
    // for (int i = 0; i < numShifts; i++){
    //     for (int j = 0; j < m_czt.m_k; j++){
    //         printf("%.6g  ", out[i*m_czt.m_k + j]);
    //     }
    //     printf("\n");
    // }

    // After the 2 loops, we perform normalisation of the output
    for (int j = 0; j < xEnergies.size(); j++)
    {
        Ipp64f normalisation = 1.0/(xEnergies.at(j) * totalGroupEnergy);
        // printf("Normalisation for shift %d is %g\n", j, normalisation);

        // Normalise by multiplying 1 / normalisation, using a complex value
        ippe::math::MulC_I(
            static_cast<Ipp32f>(normalisation),
            &out[(t_jStart + j) * m_czt.m_k], // again, this is the section for this thread
            (int)m_czt.m_k // this is the entire row
        );
    }

}

void GroupXcorrCZT::computeGroupPhaseCorrections(int t, int NUM_THREADS)
{
    // retrieve the czt object for the thread
    IppCZT32fc &m_czt = m_czts.at(t);

    // Phase correction for each frequency of CZT
    ippe::vector<Ipp64f> freq(m_czt.m_k);
    ippe::vector<Ipp64f> phase(m_czt.m_k);
    // Set the frequency slope of the CZT
    ippe::generator::Slope(
        freq.data(),
        (int)freq.size(),
        m_czt.m_f1,
        m_czt.m_fstep
    );

    ippe::vector<Ipp64fc> correction(phase.size()); // temporary 64-bit vector
    ippe::vector<Ipp64f> ones(phase.size(), 1.0);

    // Each thread works on a group
    for (int i = t; i < m_groupStarts.size(); i+=NUM_THREADS)
    {
        // Compute -2pi * f * groupStart / fs
        ippe::math::MulC(
            freq.data(),
            -IPP_2PI * (Ipp64f)m_groupStarts.at(i) / m_czt.m_fs, // remember to normalise by fs!
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

#ifdef COMPILE_FOR_PYBIND
py::array_t<float_t, py::array::c_style> GroupXcorrCZT::xcorr(
    const py::array_t<std::complex<float>, py::array::c_style> &in,
    int shiftStart, int shiftStep, int numShifts
){
    // make sure 1D
    auto buffer_info = in.request();
    if (buffer_info.shape.size() != 1)
        throw std::range_error("Input must be 1d");

    // Make the output
    py::array_t<float, py::array::c_style> out({numShifts, m_czts.at(0).m_k});

    try{
        Ipp32fc* iptr = reinterpret_cast<Ipp32fc*>(buffer_info.ptr);

        // Call the raw method
        xcorrRaw(
            iptr,
            shiftStart, shiftStep, numShifts,
            reinterpret_cast<Ipp32f*>(out.request().ptr)
        );
    }
    catch(...)
    {
        printf("Caught pybind runtime error\n");
    }

    return out;
}

void GroupXcorrCZT::addGroup(int start, 
    const py::array_t<std::complex<float>, py::array::c_style> &group,
    bool autoConj
){
    // make sure 1D
    auto buffer_info = group.request();
    if (buffer_info.shape.size()!= 1)
        throw std::range_error("Input must be 1d");

    // Call the raw addGroup
    try{
        this->addGroup(
            start, buffer_info.shape[0], 
            reinterpret_cast<Ipp32fc*>(buffer_info.ptr), 
            autoConj 
        );
    }
    catch(std::exception &e)
    {
        printf("Caught pybind runtime error: %s\n", e.what());
    }
    
}

#endif
