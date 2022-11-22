#include "GroupXcorrFFT.h"

GroupXcorrFFT::GroupXcorrFFT(
	Ipp32fc* ygroups, int numGroups, int groupLength,
	Ipp32s* offsets,
	int fs, int fftlen,
	bool autoConj
) : m_numGroups{ numGroups }, m_groupLength{ groupLength }, m_fs{ fs }
{
	m_fftlen = fftlen == -1 ? groupLength : fftlen; // set to groupLength if not specified
	if (m_fftlen < m_groupLength)
	{
		throw INVALID_FFTLEN;
	}

	// Copy the groups in
	m_ygroups.resize(m_numGroups * m_groupLength); 

	// Conjugate or copy the complex groups of data
	if (autoConj)
	{
		ippsConj_32fc(ygroups, m_ygroups.data(), (int)m_ygroups.size());
	}
	else
	{
		ippsCopy_32fc(ygroups, m_ygroups.data(), (int)m_ygroups.size());
	}
	// Calculate the norm sq here
	ippsNorm_L2_32fc64f(m_ygroups.data(), (int)m_ygroups.size(), &m_ygroupsNormSq);
	m_ygroupsNormSq = m_ygroupsNormSq * m_ygroupsNormSq;

	// Copy the offsets
	m_offsets.resize(numGroups);
	// Deduct the first value to zero it while copying
	ippsSubC_32s_Sfs((Ipp32s*)offsets, offsets[0], m_offsets.data(), (int)m_offsets.size(), 0);

	// Compute the group phases
	calculateGroupPhases();
}

void GroupXcorrFFT::makeFreq(ippe::vector<Ipp64f>& fftfreq)
{
	fftfreq.resize(m_fftlen);

	Ipp64f tmp;
	for (int i = 0; i < m_fftlen; i++)
	{
		tmp = (double)i * (double)m_fs / (double)m_fftlen;
		fftfreq.at(i) = tmp >= (double)m_fs / 2.0 ? tmp - (double)m_fs : tmp;
	}
}

void GroupXcorrFFT::calculateGroupPhases()
{
	//// Calculate the fft freq vector first
	//ippe::vector<Ipp64f> fftfreq;
	//makeFreq(fftfreq);

	// Compute the group phases
	ippe::vector<Ipp64fc> temp_64fc(m_fftlen);
	Ipp64f phase, rFreq, rFreqFractional;

	m_groupPhases.resize(m_fftlen * m_offsets.size());
	for (int i = 0; i < m_offsets.size(); i++)
	{
		phase = 0;
		rFreq = (Ipp64f)m_offsets.at(i) / (Ipp64f)m_fftlen;
		rFreq = modf(rFreq, &rFreqFractional); // but we don't need the fractional part
		rFreq = rFreq != 0 ? 1.0 - rFreq : rFreq; // this is to effectively have -rFreq which is needed, but only if its non-zero
		//printf("rFreq[%d] = %f\n", i, rFreq); // debugging
		ippsTone_64fc(temp_64fc.data(), m_fftlen, 1.0, rFreq, &phase, ippAlgHintAccurate); // generate at 64fc for higher precision
		// Convert from 64f to 32f
		ippsConvert_64f32f((Ipp64f*)temp_64fc.data(), (Ipp32f*)&m_groupPhases.at(i * m_fftlen), 2 * m_fftlen); // *2 length because complex
	}
}

void GroupXcorrFFT::xcorr(const Ipp32fc *rx, const int rxlen, const int* shifts, const int shiftslen, Ipp32f* out, int NUM_THREADS)
{
	std::vector<std::thread> threads(NUM_THREADS);

	for (int t = 0; t < NUM_THREADS; t++)
	{
		threads.at(t) = std::thread(
			&GroupXcorrFFT::xcorr_thread,
			this,
			t, NUM_THREADS,
			rx, rxlen,
			shifts, shiftslen,
			out
		);
	}

	for (int t = 0; t < NUM_THREADS; t++)
	{
		threads.at(t).join();
	}
}

void GroupXcorrFFT::xcorr_thread(int thrdIdx, int NUM_THREADS, const Ipp32fc* rx, const int rxlen, const int* shifts, const int shiftslen, Ipp32f *out)
{
	// Allocate thread workspace
	ippe::vector<Ipp32fc> pdt;
	ippe::vector<Ipp32fc> pdtfft;
	ippe::vector<Ipp64f> rxgroupNormSqCollect;
	ippe::vector<Ipp32fc> pdtfftcombined;
	Ipp64f rxgroupNormSq;
	ippe::vector<Ipp32f> qf2(m_fftlen);
	// FFT related workspace
	int sizeSpec = 0, sizeInit = 0, sizeBuf = 0;
	ippsDFTGetSize_C_32fc(m_fftlen, IPP_FFT_NODIV_BY_ANY, ippAlgHintNone, &sizeSpec, &sizeInit, &sizeBuf);
	/* memory allocation */
	IppsDFTSpec_C_32fc* pDFTSpec = (IppsDFTSpec_C_32fc*)ippMalloc(sizeSpec); 
	Ipp8u* pDFTBuffer = (Ipp8u*)ippMalloc(sizeBuf);
	Ipp8u* pDFTMemInit = (Ipp8u*)ippMalloc(sizeInit);
	ippsDFTInit_C_32fc(m_fftlen, IPP_FFT_NODIV_BY_ANY, ippAlgHintNone, pDFTSpec, pDFTMemInit); 

	// Loop over values
	//printf("DEBUG: Beginning loop over shifts\n");
	int shift;
	for (int i = thrdIdx; i < shiftslen; i += NUM_THREADS)
	{
		// Get the current shift index
		shift = shifts[i];
		//printf("DEBUG: shift %d\n", i);

		// Perform the dot and fft over the groups
		dot_and_fft(rx, shift, pdt, pdtfft, rxgroupNormSqCollect, pdtfftcombined, &rxgroupNormSq, pDFTSpec, pDFTBuffer);
		//printf("DEBUG: dot_and_fft complete\n");
		//printf("DEBUG: rxgroupNormSq = %f\n", rxgroupNormSq);


		// Calculate abs squared of the final output
		ippsPowerSpectr_32fc(pdtfftcombined.data(), qf2.data(), (int)pdtfftcombined.size());
		//printf("DEBUG: ippsPowerSpectr_32fc complete\n");
		//if (i == 0)
		//{
			//printf("DEBUG at 0: %f %f, %f %f\n\n", pdtfftcombined.at(0).re, pdtfftcombined.at(0).im, pdtfftcombined.at(1).re, pdtfftcombined.at(1).im);
			//printf("DEBUG at 0: %f, %f\n", qf2.at(0), qf2.at(1));
		//}

		// Scale by the energies of both inputs
		ippsDivC_32f(qf2.data(), static_cast<Ipp32f>(rxgroupNormSq * m_ygroupsNormSq), &out[i * m_fftlen], (int)m_fftlen);
		//printf("DEBUG: ippsDivC_32f complete\n");
	}

	// Cleanup
	ippFree(pDFTSpec);
	ippFree(pDFTBuffer);
	ippFree(pDFTMemInit);
}

void GroupXcorrFFT::dot_and_fft(
	const Ipp32fc* rx,
	int shift,
	ippe::vector<Ipp32fc> &pdt,
	ippe::vector<Ipp32fc> &pdtfft,
	ippe::vector<Ipp64f> &rxgroupNormSqCollect,
	ippe::vector<Ipp32fc> &pdtfftcombined,
	Ipp64f *rxgroupNormSq,
	IppsDFTSpec_C_32fc *pDFTSpec,
	Ipp8u *pDFTBuffer)
{
	// resize if needed
	pdt.resize(m_fftlen);
	pdtfft.resize(m_fftlen);
	rxgroupNormSqCollect.resize(m_numGroups);
	pdtfftcombined.resize(m_fftlen);

	// Zero the final output
	ippsZero_32fc(pdtfftcombined.data(), (int)pdtfftcombined.size());

	// Loop over the groups
	for (int g = 0; g < m_numGroups; g++)
	{
		// Calculate the norm
		ippsNorm_L2_32fc64f(&rx[shift + m_offsets.at(g)], m_groupLength, &rxgroupNormSqCollect.at(g));
		// Remember to square it
		rxgroupNormSqCollect.at(g) = rxgroupNormSqCollect.at(g) * rxgroupNormSqCollect.at(g);
		//printf("DEBUG: dot_and_fft norm complete\n");

		// Multiply ygroup with rxgroup
		ippsMul_32fc(&rx[shift + m_offsets.at(g)], &m_ygroups.at(g * m_groupLength), pdt.data(), (int)m_groupLength);
		//printf("DEBUG: dot_and_fft ippsMul_32fc complete\n");

		// FFT the product (produce pdtfft)
		ippsDFTFwd_CToC_32fc(pdt.data(), pdtfft.data(), pDFTSpec, pDFTBuffer);
		//printf("DEBUG: dot_and_fft ippsDFTFwd_CToC_32fc complete\n");
		//printf("DEBUG GROUP %d: pdtfft -> %f %f, %f %f \n", g, pdtfft.at(0).re, pdtfft.at(0).im, pdtfft.at(1).re, pdtfft.at(1).im);

		// Accumulate into final vector with the phase correction
		ippsAddProduct_32fc(&m_groupPhases.at(g * m_fftlen), pdtfft.data(), pdtfftcombined.data(), m_fftlen);
		//printf("DEBUG: dot_and_fft ippsAddProduct_32fc complete\n");
		//printf("DEBUG GROUP %d: pdtfftcombined -> %f %f, %f %f \n", g, pdtfftcombined.at(0).re, pdtfftcombined.at(0).im, pdtfftcombined.at(1).re, pdtfftcombined.at(1).im);

	}

	// Remember to sum up all the rxgroupNormSqs
	ippsSum_64f(rxgroupNormSqCollect.data(), (int)rxgroupNormSqCollect.size(), rxgroupNormSq);
}



GroupXcorrFFT::~GroupXcorrFFT()
{

}