#pragma once

#include "../ippCZT/CZT.h"
#include "../../ipp_ext/include/ipp_ext.h"
#include <vector>
#include <iostream>

// This class works specifically on 32fc inputs.
class GroupXcorrCZT
{
public:
    GroupXcorrCZT(){}
    GroupXcorrCZT(int maxlen, double f1, double f2, double fstep, double fs)
        : m_czt{maxlen, f1, f2, fstep, fs}
    {}

    void addGroup(int start, int length, Ipp32fc *group, bool autoConj=true);
    void resetGroups();

    void xcorr(Ipp32fc *x, int shiftStart, int shiftStep, int numShifts, Ipp32f *output);

    int getCZTdimensions(){ return m_czt.m_k; }

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

    IppCZT32fc m_czt;

    void computeGroupPhaseCorrections();
};
