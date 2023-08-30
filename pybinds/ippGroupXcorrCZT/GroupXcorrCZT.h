#pragma once

#include "../ippCZT/CZT.h"
#include "../../ipp_ext/include/ipp_ext.h"
#include <vector>
#include <iostream>
#include <thread>

// This class works specifically on 32fc inputs.
class GroupXcorrCZT
{
public:
    GroupXcorrCZT(){}
    GroupXcorrCZT(int maxlen, double f1, double f2, double fstep, double fs, size_t NUM_THREADS=1)
        : m_threads{NUM_THREADS}, m_czts{NUM_THREADS}
    {
        if (NUM_THREADS < 1) throw std::invalid_argument("Number of threads must be greater than 0");
        for (auto czt : m_czts)
            czt = IppCZT32fc(maxlen, f1, f2, fstep, fs); // instantiate the vector of CZTs
    }

    void addGroup(int start, int length, Ipp32fc *group, bool autoConj=true);
    void resetGroups();

    void xcorr(
        Ipp32fc *x, 
        int shiftStart, int shiftStep, int numShifts, 
        Ipp32f *out
    );

    int getCZTdimensions(){ return m_czts.at(0).m_k; }

    // Debugging?
    void printGroups(){
        for (int i=0; i < m_groups.size(); i++){
            printf("Group %d: [%d, %d)\n", 
                i, m_groupStarts[i], 
                m_groupStarts[i] + (int)m_groups[i].size());
            printf("Energy = %f\n", m_groupEnergies[i]);
        }
    }

private:
    std::vector<int> m_groupStarts;
    std::vector<ippe::vector<Ipp32fc>> m_groups;
    ippe::vector<Ipp64f> m_groupEnergies;
    std::vector<ippe::vector<Ipp32fc>> m_groupPhaseCorrections;

    std::vector<std::thread> m_threads;

    std::vector<IppCZT32fc> m_czts;

    // These are the main runtime methods invoked by xcorr(), in order
    void computeGroupPhaseCorrections(int t=0, int NUM_THREADS=1);
    void correlateGroups(
        Ipp32fc *x, 
        int shiftStart, int shiftStep, int numShifts,
        Ipp64f totalGroupEnergy,
        Ipp32f *out,
        int t=0, int NUM_THREADS=1
    );
    
};
