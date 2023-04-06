#pragma once

#include "ipp.h"
#include "ipp_ext.h"
#include <vector>
#include <thread>

// Container to perform the standard sliding product and fft (IPP's DFT)
// Similar to the other things I had in commonMex previously
class IppXcorrFFT_32fc
{
public:
	IppXcorrFFT_32fc(const Ipp32fc* cutout, int cutoutlen, int num_threads, bool autoConj);
	~IppXcorrFFT_32fc();

	// main runtime method
	void xcorr(
		const Ipp32fc* src,
		const int srclen,
		const int startIdx, 
		const int endIdx, 
		const int idxStep
	);

	// output vectors
	std::vector<float> m_productpeaks;
	std::vector<int> m_freqlistinds;

private:
	int m_cutoutlen;
	int m_num_threads = 1;

    ippe::vector<Ipp32fc> m_cutout;
	Ipp32f m_cutoutNormSq;

	// workspace
	std::vector<std::thread> m_threads;
    std::vector<ippe::DFTCToC<Ipp32fc>> m_ffts;
	std::vector<ippe::vector<Ipp32fc>> m_work_32fc_1;
	std::vector<ippe::vector<Ipp32fc>> m_work_32fc_2;

	// internal work method
	void xcorr_thread(
		const Ipp32fc* src,
		const int srclen,
		const int startIdx, 
		const int endIdx, 
		const int idxStep,
		const int tIdx
	);

};