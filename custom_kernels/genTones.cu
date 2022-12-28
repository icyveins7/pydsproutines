#include <cupy/complex.cuh>

// this is the naive implementation, we just calculate the exact exponential for each sample
extern "C" __global__
void genTonesDirect_64f(
	const double f0,
	const double fstep,
	const int numFreqs,
	const int len,
	complex<double> *out)
{
	// spawn as many blocks as required to fulfill the length
	// each block just writes to its own section of the output, iterates over every frequency
	int startidx = blockIdx.x * blockDim.x;
	int i = threadIdx.x + startidx; // each thread will work on this sample, for every frequency
	int offset;
	
	double f, phase, re, im;
	
	for (int fidx = 0; fidx < numFreqs; fidx++)
	{
		f = f0 + fidx * fstep; // the frequency we are working on
		offset = fidx * len; // offset to the row to write to
		sincospi(2 * f * (double)i, &im, &re); // write the components to stack
		if (i < len){ // coalesced global writes
			out[offset + i] = complex<double>(re, im);
		}
	}
}

// this doesn't calculate for each frequency, but just multiplies the original values by a constant complex number
// there might be some loss due to computational error build up over many frequencies
// note that this is actually slower!
extern "C" __global__
void genTonesScaling_64f(
	const double f0,
	const double fstep,
	const int numFreqs,
	const int len,
	complex<double> *out)
{
	// spawn as many blocks as required to fulfill the length
	// each block just writes to its own section of the output, iterates over every frequency
	int startidx = blockIdx.x * blockDim.x;
	int i = threadIdx.x + startidx; // each thread will work on this sample, for every frequency
	int offset = 0;
	
	double f, phase, re, im;
	complex<double> outstack;
	
	// calculate the value for the first frequency
	sincospi(2 * f0 * (double)i, &im, &re);
	outstack = complex<double>(re, im); // we keep a copy on stack
	if (i < len){ // coalesced global writes
		out[offset + i] = outstack;
	}
	
	// now calculate the complex number to scale by
	sincospi(2 * fstep * (double)i, &im, &re);
	complex<double> alpha = complex<double>(re, im);
	
	// loop over the rest of the frequencies
	for (int fidx = 1; fidx < numFreqs; fidx++) // now start from 1
	{
		f = f0 + fidx * fstep; // the frequency we are working on
		offset = fidx * len; // offset to the row to write to
		outstack = outstack * alpha;
		if (i < len){
			out[offset + i] = outstack;
		}
	}
}