#pragma once

#include "../ippCZT/CZT.h"
#include "../../ipp_ext/include/ipp_ext.h"
#include <vector>
#include <iostream>
#include <thread>

// =====================================
#ifdef COMPILE_FOR_PYBIND
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
#endif
// =====================================

// This class works specifically on 32fc inputs.
class GroupXcorrCZT
{
public:
    GroupXcorrCZT(){}
    GroupXcorrCZT(int maxlen, double f1, double f2, double fstep, double fs)
        : m_threads{1}
    {
        m_czts.emplace_back(maxlen, f1, f2, fstep, fs);
    }
    GroupXcorrCZT(int maxlen, double f1, double f2, double fstep, double fs, size_t NUM_THREADS)
        : m_threads{NUM_THREADS}//, m_czts{NUM_THREADS}
    {
        if (NUM_THREADS < 1) throw std::invalid_argument("Number of threads must be greater than 0");
        for (size_t i = 0; i < NUM_THREADS; ++i)
            m_czts.emplace_back(maxlen, f1, f2, fstep, fs); // instantiate the vector of CZTs
    }

    /// @brief Adds a group to use for correlation.
    /// @param start Defines the relative start index of the group.
    /// @param length Number of elements in the group.
    /// @param group Pointer to the group data. Data will be copied starting from this pointer.
    /// @param autoConj Enables auto-conjugation of the group. Default is true.
    void addGroup(int start, int length, Ipp32fc *group, bool autoConj=true);

    /// @brief Adds groups from an array via slices specified by start indices and lengths.
    /// This helps to automatically zero the earliest index and track it for each group.
    /// For example, if [start, start+length) are [10, 20) and [30, 42) then
    /// Group 1: start = 0, Group 2: start = 20 (offset from 10).
    /// @param starts The start indices from 'arr' to slice.
    /// @param lengths The lengths of each group from 'arr' to slice.
    /// @param arr The long array to slice from.
    /// @param autoConj Enables auto-conjugation of each group.
    void addGroupsFromArray(int *starts, int *lengths, size_t numGroups, Ipp32fc *arr, bool autoConj=true);

    /// @brief Clears all groups and their associated internal data.
    void resetGroups();

    void xcorrRaw(
        Ipp32fc *x, 
        int shiftStart, int shiftStep, int numShifts, 
        Ipp32f *out, int xLength=-1
    );

    // Some getters
    int getCZTdimensions(){ return m_czts.at(0).m_k; }
    size_t getNumThreads(){ return m_threads.size(); }

    // Debugging?
    void printGroups(){
        for (int i=0; i < (int)m_groups.size(); i++){
            printf("Group %d: [%d, %d)\n", 
                i, m_groupStarts[i], 
                m_groupStarts[i] + (int)m_groups[i].size());
            printf("Energy = %f\n", m_groupEnergies[i]);
        }
    }

    #ifdef COMPILE_FOR_PYBIND
    void addGroup(int start,
        const py::array_t<std::complex<float>, py::array::c_style> &group,
        bool autoConj=true
    );

    void addGroupsFromArray(
        const py::array_t<int, py::array::c_style> &starts,
        const py::array_t<int, py::array::c_style> &lengths,
        const py::array_t<std::complex<float>, py::array::c_style> &arr,
        bool autoConj=true
    );

    py::array_t<float_t, py::array::c_style> xcorr(
        const py::array_t<std::complex<float>, py::array::c_style> &in,
        int shiftStart, int shiftStep, int numShifts
    );
    #endif

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
        int xLength,
        int t=0, int NUM_THREADS=1
    );
    
};
