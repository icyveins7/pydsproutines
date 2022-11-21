#pragma once
#include "ipp.h"
#include "ipp_ext.h"
#include <thread>
#include <vector>

#define INVALID_FFTLEN 1

class GroupXcorrFFT
{
public:
	GroupXcorrFFT(
		Ipp32fc *ygroups, int numGroups, int groupLength,
		int *offsets,
		int fs, int fftlen=-1,
		bool autoConj=true
	);
	~GroupXcorrFFT();

	void xcorr(const Ipp32fc* rx, const int rxlen, const int* shifts, const int shiftslen, Ipp32f* out, int NUM_THREADS = 1);

	// getters
	int getFftlen() { return m_fftlen; }
	ippe::vector<Ipp32fc>& getGroupPhases() { return m_groupPhases; }

private:
	// Computation methods
	void makeFreq(ippe::vector<Ipp64f>& fftfreq);
	void calculateGroupPhases();
	void xcorr_thread(int thrdIdx, int NUM_THREADS, const Ipp32fc* rx, const int rxlen, const int* shifts, const int shiftslen, Ipp32f* out);
	void dot_and_fft(
		const Ipp32fc* rx,
		int shift,
		ippe::vector<Ipp32fc>& pdt,
		ippe::vector<Ipp32fc>& pdtfft,
		ippe::vector<Ipp64f>& rxgroupNormSqCollect,
		ippe::vector<Ipp32fc>& pdtfftcombined,
		Ipp64f* rxgroupNormSq,
		IppsDFTSpec_C_32fc* pDFTSpec,
		Ipp8u* pDFTBuffer);

	// Member variables
	ippe::vector<Ipp32fc> m_ygroups;
	Ipp64f m_ygroupsNormSq;
	std::vector<int> m_offsets;
	ippe::vector<Ipp32fc> m_groupPhases;

	int m_numGroups;
	int m_groupLength;
	int m_fs;
	int m_fftlen;
};