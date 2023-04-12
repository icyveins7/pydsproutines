#include "IppXcorrFFT.h"

int IppXcorrFFT_32fc::getOutputLength(
	const int startIdx,
	const int endIdx,
	const int idxStep
){
	int length = (endIdx - startIdx) / idxStep;
	if ((endIdx - startIdx) % idxStep != 0)
	    length += 1;
	return length;
}

void IppXcorrFFT_32fc::xcorr(
	const Ipp32fc* src,
	const int srclen,
	const int startIdx, 
	const int endIdx, 
	const int idxStep
){
	// resize the outputs accordingly
	int outputlen = getOutputLength(startIdx, endIdx, idxStep);
	m_productpeaks.resize(outputlen);
	m_freqlistinds.resize(outputlen);

	// start threads to iterate over
	m_threads.resize(m_num_threads);
	for (int i = 0; i < m_num_threads; i++)
	{
		// printf("Launching thread %d\n", i);
		m_threads[i] = std::thread(
			&IppXcorrFFT_32fc::xcorr_thread,
			this, 
			src, 
			srclen, 
			startIdx, 
			endIdx, 
			idxStep,
			i, // thread id
			m_productpeaks.data(),
			m_freqlistinds.data()
		);
	}

	// wait for all threads to finish
    for (int i = 0; i < m_num_threads; i++)
	{
		m_threads[i].join();
	}
}

void IppXcorrFFT_32fc::xcorr_array(
	const Ipp32fc* src,
	const int srclen,
	const int startIdx, 
	const int endIdx, 
	const int idxStep,
	float *productpeaks,
	int *freqlistinds,
	int outputlength
){
	// check the output length is 'correct' as a validation mechanic
	if (getOutputLength(startIdx, endIdx, idxStep) != outputlength)
	{
		throw std::runtime_error("Output length is not correct");
	}

	// start threads to iterate over
	m_threads.resize(m_num_threads);
	for (int i = 0; i < m_num_threads; i++)
	{
		// printf("Launching thread %d\n", i);
		m_threads[i] = std::thread(
			&IppXcorrFFT_32fc::xcorr_thread,
			this, 
			src, 
			srclen, 
			startIdx, 
			endIdx, 
			idxStep,
			i, // thread id
			productpeaks,
            freqlistinds
		);
	}

	// wait for all threads to finish
    for (int i = 0; i < m_num_threads; i++)
	{
		m_threads[i].join();
	}
}

void IppXcorrFFT_32fc::xcorr_thread(
    const Ipp32fc* src,
    const int srclen,
    const int startIdx,
	const int endIdx,
    const int idxStep,
    const int tIdx,
	float *productpeaks,
    int *freqlistinds
){
	int outputlen = getOutputLength(startIdx, endIdx, idxStep);

	Ipp32f maxval;
	int maxind;
	Ipp64f slicenorm;

	// local fft object
	ippe::DFTCToC<Ipp32fc> fftobj((size_t)m_cutoutlen);
	// and local workspace
	ippe::vector<Ipp32fc> work_32fc_1(m_cutoutlen);
	ippe::vector<Ipp32fc> work_32fc_2(m_cutoutlen);

    int i;
	for (int t = tIdx; t < outputlen; t += m_num_threads)
	{
		// Define the accessor index for the src
		i = startIdx + t * idxStep; 

		// printf("In thread %d, startIdx %d, on output %d/%d\n", tIdx, i, t, m_productpeaks.size());

		// Don't compute if we're out of range
		if (i < 0 || i + m_cutoutlen > srclen)
		{
			productpeaks[t] = 0.0f;
			freqlistinds[t] = 0;
			continue;
		}

		// First we multiply, use the first workspace
		ippsMul_32fc(
			m_cutout.data(),
			&src[i],
			work_32fc_1.data(),
			m_cutoutlen
		);
		// printf("Completed Mul\n");

		// Then we fft the output, use the second workspace
		try{
			// m_ffts[tIdx].fwd(
			fftobj.fwd(
				work_32fc_1.data(),
				work_32fc_2.data()
			);
		}
		catch(const std::exception& e){
			printf("Exception for thread %d: %s\n", tIdx, e.what());
			// printf("workspace size: %zd\n", m_work_32fc_1[tIdx].size());
			// printf("workspace size: %zd\n", m_work_32fc_2[tIdx].size());
		}
		// printf("Completed FFT\n");

		// Get abs squared, reuse first workspace
		ippsPowerSpectr_32fc(
			work_32fc_2.data(),
			(Ipp32f*)work_32fc_1.data(), // note that this uses the 'first half' of the alloc'ed memory
			m_cutoutlen
		);

		// Get the max index, and the associated value
		ippsMaxIndx_32f(
			(Ipp32f*)work_32fc_1.data(),
			m_cutoutlen,
			&maxval, &maxind
		);

		// get the norm sq for this slice
		ippsNorm_L2_32fc64f(&src[i], m_cutoutlen, &slicenorm);

		// compute the output with scaling
		productpeaks[t] = maxval / m_cutoutNormSq / (Ipp32f)(slicenorm * slicenorm);
		freqlistinds[t] = maxind;

	}
}

/////////////////////////// CONSTRUCTOR AND DESTRUCTOR
IppXcorrFFT_32fc::IppXcorrFFT_32fc(const Ipp32fc* cutout, int cutoutlen, int num_threads, bool autoConj)
	: m_cutoutlen{cutoutlen}, m_num_threads{num_threads}
{
	// Store cutout internally
	m_cutout.resize(m_cutoutlen);
	ippsCopy_32fc(cutout, m_cutout.data(), m_cutoutlen);
	if (autoConj)
		ippsConj_32fc_I(m_cutout.data(), m_cutoutlen);

	// compute the norm squared and store it
	Ipp64f norm2;
	ippsNorm_L2_32fc64f(m_cutout.data(), m_cutoutlen, &norm2);
	m_cutoutNormSq = static_cast<Ipp32f>(norm2*norm2);
}

IppXcorrFFT_32fc::~IppXcorrFFT_32fc()
{

}