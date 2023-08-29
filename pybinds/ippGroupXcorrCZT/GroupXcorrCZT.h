#pragma once

#include "../ippCZT/CZT.h"
#include "../../ipp_ext/include/ipp_ext.h"
#include <vector>

class GroupXcorrCZT
{
public:
    GroupXcorrCZT(){}
    GroupXcorrCZT(int maxlen, double f1, double f2, double fstep, double fs)
        : m_czt{maxlen, f1, f2, fstep, fs}
    {}

    void addGroup(int start, int length, Ipp32fc *group, bool autoConj=true);
    void resetGroups();

    void xcorr(Ipp32fc *x, int shiftStart, int shiftStep, int numShifts, Ipp32fc *output);

private:
    std::vector<int> m_groupStarts;
    std::vector<int> m_groupLengths;
    std::vector<ippe::vector<Ipp32fc>> m_groups;
    ippe::vector<Ipp64f> m_groupEnergies;
    std::vector<ippe::vector<Ipp32fc>> m_groupPhaseCorrections;
    ippe::vector<Ipp64f> m_xEnergies;

    IppCZT32fc m_czt;

    void computeGroupPhaseCorrections();
};
