/*
Using nvprof to examine the kernel times shows that the kernels are actually slower as compared to 
the natural cupy exp(...) calls. However, the cupy calls are split into several steps, with significant 
downtime between each kernel call, which result in a slower process when the size is relatively small.
*/

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
// note that even though python timing may show this to be slower, the kernel timing for this is around 10x faster than the direct one
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

// ============================== 32f versions ======================================================
// note that internally it's still computed with doubles, but we write to global mem as floats
// as such, the extra explicit casts actually make it slightly slower than the 64f version
extern "C" __global__
void genTonesDirect_32f(
	const double f0,
	const double fstep,
	const int numFreqs,
	const int len,
	complex<float> *out)
{
	// spawn as many blocks as required to fulfill the length
	// each block just writes to its own section of the output, iterates over every frequency
	int startidx = blockIdx.x * blockDim.x;
	int i = threadIdx.x + startidx; // each thread will work on this sample, for every frequency
	int offset;
	
	double f, phase;
    double re, im;
	
	for (int fidx = 0; fidx < numFreqs; fidx++)
	{
		f = f0 + fidx * fstep; // the frequency we are working on
		offset = fidx * len; // offset to the row to write to
		sincospi(2 * f * (double)i, &im, &re); // write the components to stack
		if (i < len){ // coalesced global writes
			out[offset + i] = complex<float>(re, im);
		}
	}
}

// again, computed with doubles, saved as floats
extern "C" __global__
void genTonesScaling_32f(
	const double f0,
	const double fstep,
	const int numFreqs,
	const int len,
	complex<float> *out)
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
			out[offset + i] = complex<float>(outstack.real(), outstack.imag());
		}
	}
}

/* 
What about dot producting tones directly? 
This would be like CZTs, except possibly without the overhead of multiple FFT calls and optimised shared mem usage?
Benefit of this is to read the source only once from global memory.
*/

// 1. We use shared memory to store up to 64 * 64 complex64 values of src*tone. To keep things simple, we can fix this size, along with the kernel parameters, as 64x64 always.
// 2. Then we sum up within shared memory and then output to an external array (this will then require a separate second kernel to sum results together)
// 3. Return to step 1 with the rest of the frequencies, until all frequencies are complete.
extern "C" __global__
void dotTonesScaling_32f(
	const double f0,
	const double fstep,
	const int numFreqs,
	const int len,
	const complex<float> *src,
	complex<float> *out)
{
   // spawn as many blocks as required to fulfill the length
	// each block just writes to its own section of the output, iterates over every frequency
	int startidx = blockIdx.x * blockDim.x;
	int i = threadIdx.x + startidx; // each thread will work on this sample, for every frequency

   // initialise shared memory
   extern __shared__ double s[];   
   complex<float> *s_ws = (complex<float>*)s; // (64*64) complex floats
   
   // let's have a variable to mark the row in shared memory we are currently writing to
   int s_row;

   // the rest of the variable declarations as before
	double f, phase, re, im;
	complex<double> outstack;
	
	// calculate the value for the first frequency
	sincospi(2 * f0 * (double)i, &im, &re);
	outstack = complex<double>(re, im); // we keep a copy on stack
	
	// now calculate the complex number to scale by
	sincospi(2 * fstep * (double)i, &im, &re);
	complex<double> alpha = complex<double>(re, im);
    
    // make a double version of the global mem source
    complex<double> src64f;
	
	// no point doing anything if this thread is beyond the length
	if (i < len)
	{
    	// loop over all the frequencies
    	for (int fidx = 0; fidx < numFreqs; fidx++)
    	{
        	// define the row in shared memory we will be working on
        	s_row = fidx % 64;
        	
        	// for everything other than the first loop, we must multiply by the tone
        	if (fidx > 0)
        	{
        		outstack = outstack * alpha; 
        	}
        	else // for the first loop, we instead now multiply by the source (this is the only time we read the source)
        	{
                src64f = complex<double>(src[i].real(), src[i].imag());
            	outstack = outstack * src64f;
        	}

        	// after we have done the multiply of the tone, we save it in the appropriate spot in the workspace on shared mem
            s_ws[64 * s_row + threadIdx.x] = complex<float>(outstack.real(), outstack.imag()); // cast to floats just like before
            
            // finally, if we have completed a batch of 64 (or we're on the last freq),
            // it's time to sum up in sharedmem and output to global mem after
            if (fidx % 64 == 0 & fidx != 0 || fidx == numFreqs-1)
            {
                // to be completed
            }
            
          
    	}
	}
}

