#include <cupy/complex.cuh>

// This kernel only loads the filter taps into shared memory, without an extra workspace
// This is useful if the number of filter taps is very long, and hence leaves less than 2*length of complex workspace for the data to sit in.
// Relevant only for real-only filter taps.
// In order to not waste warps, it is recommended to set outputPerBlk to a multiple of blockDim.
extern "C" __global__
void filter_smtaps(
    const complex<float> *d_x, const int len,
    const float *d_taps, const int tapslen,
    const int outputPerBlk,
    complex<float> *d_out, int outlen,
    const complex<float> *d_delay, const int delaylen,
    const int dsr, const int dsPhase)
{
    // allocate shared memory
    extern __shared__ double s[];
    
    float *s_taps = (float*)s; // (tapslen) floats
    /* Tally:  */

    // load shared memory
    for (int t = threadIdx.x; t < tapslen; t = t + blockDim.x){
        s_taps[t] = d_taps[t];
    }
    
    __syncthreads();
    
    // Begin computations
    int i; // output index
    int k; // reference index (not equal to output if downsample is >1)
    complex<float> z; // stack-var for each thread
    for (int t = threadIdx.x; t < outputPerBlk; t = t + blockDim.x)
    {
        z = 0; // reset before the output

        i = blockIdx.x * outputPerBlk + t; // This is the thread's output index
        k = i * dsr + dsPhase; // This is the reference index

        // Exit if we hit the end
        if (i >= outlen)
            break;

        // Otherwise loop over the taps
        for (int j = 0; j < tapslen; j++)
        {
            int xIdx = k - j;

            // accumulate
            if (xIdx >= 0 && xIdx < len)
                z = z + d_x[xIdx] * s_taps[j]; // this uses the input data
            else if (delaylen + xIdx >= 0 && d_delay != NULL) // d_delay must be supplied for this to work
                z = z + d_delay[delaylen + xIdx] * s_taps[j]; // this uses the delay data (from previous invocations)
        }

        // Coalesced writes
        d_out[i] = z;
    }
 
}


// ================
// If the number of taps is small, we can allocate a workspace for the complex-valued inputs
// and then use that workspace to prevent repeated global reads of the same element
extern "C" __global__
void filter_smtaps_sminput(
    const complex<float> *d_x, const int len,
    const float *d_taps, const int tapslen,
    const int outputPerBlk,
    const int workspaceSize, // this must correspond to outputPerBlk + tapslen - 1
    complex<float> *d_out, int outlen)
{
    // allocate shared memory
    extern __shared__ double s[];
    
    float *s_taps = (float*)s; // (tapslen) floats
    complex<float> *s_ws = (complex<float>*)&s_taps[tapslen]; // workspaceSize
    /* Tally:  */

    // load shared memory taps
    for (int t = threadIdx.x; t < tapslen; t = t + blockDim.x){
        s_taps[t] = d_taps[t];
    }
    // load the shared memory workspace
    int i0 = blockIdx.x * outputPerBlk; // this is the first output index
    int workspaceStart = i0 - tapslen + 1; // this is the first index that is required
    // int workspaceEnd   = i0 + outputPerBlk; // this is the last index that is required (non-inclusive)
    int i;
    for (int t = threadIdx.x; t < workspaceSize; t = t + blockDim.x)
    {
        i = workspaceStart + t; // this is the input source index to copy
        if (i < 0 || i >= outlen) // set to 0 if its out of range
            s_ws[t] = 0;
        else
            s_ws[t] = d_x[i];
    }
    
    __syncthreads();
    
    // Begin computations
    complex<float> z; // stack-var for each thread
    int wsi;
    for (int t = threadIdx.x; t < outputPerBlk; t = t + blockDim.x)
    {
        z = 0; // reset before the output

        i = blockIdx.x * outputPerBlk + t; // This is the output index
        wsi = tapslen - 1 + t; // this is the 'equivalent' source index from shared memory

        // Exit if we hit the end
        if (i >= outlen)
            break;

        // Otherwise loop over the taps and the shared mem workspace
        for (int j = 0; j < tapslen; j++)
        {
            // accumulate
            z = z + s_ws[wsi - j] * s_taps[j];
        }

        // Coalesced writes
        d_out[i] = z;
    }
 
}


/*
Allocate a grid with (x,y) blocks such that gridDim.y == numRows.
Each thread computes 1 output.

Naive implementation:
Each thread sums over its own window, and thread windows overlap.
E.g.
Thread 0: 0 -> N-1
Thread 1: 1 -> N
Thread 2: 2 -> N+1
...
Each block will have N threads, which works on N outputs of a particular row.
*/
extern "C" __global__
void multiMovingAverage(
    const float *d_x,
    const int numRows, const int numCols, // same dimensions for d_x and d_out
    const int avgLength, // moving average window size,
    float *d_out
){
    // allocate shared memory
    extern __shared__ double s[];
    
    float *s_x = (float*)s; // (tapslen) floats
    /* Tally:  */

    // Calculate this block's ROW offset
    int blockOffset = blockIdx.y * numCols;
    // Calculate shared memory usage (each thread computes 1 output)
    int sharedMemSize = blockDim.x + avgLength - 1;
    // Determine the first index of this block's input row to start copying
    int i0 = blockIdx.x*blockDim.x - avgLength + 1; // this may be negative!

    // Copy the shared memory, setting 0s if it references a negative index
    for (int t = threadIdx.x; t < sharedMemSize; t += blockDim.x)
    {
        // Evaluate the column index for this block to read from
        int idx = t + i0;
        if (idx >= 0 && idx < numCols)
            s_x[t] = d_x[blockOffset + idx];
        else
            s_x[t] = 0;
    } 
   
    __syncthreads();

    // Now compute
    for (int t = threadIdx.x; t < blockDim.x; t += blockDim.x)
    {
        // No writing out of bounds
        if (blockIdx.x*blockDim.x + t < numCols)
        {
            double sum = 0.0;

            for (int i = 0; i < avgLength; i++)
            {
                sum += s_x[t+i];
            }

            d_out[blockOffset + blockIdx.x*blockDim.x + t] = (float)(sum / (double)avgLength);
    
        }
   }
    

}


