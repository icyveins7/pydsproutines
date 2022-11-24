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
    complex<float> *d_out, int outlen)
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
    complex<float> z; // stack-var for each thread
    for (int t = threadIdx.x; t < outputPerBlk; t = t + blockDim.x)
    {
        z = 0; // reset before the output

        i = blockDim.x * outputPerBlk + t; // This is the output index

        // Exit if we hit the end
        if (i >= outlen)
            break;

        // Otherwise loop over the taps
        for (int j = 0; j < tapslen; j++)
        {
            // accumulate
            z = d_x[i - j] * s_taps[j];
        }

        // Coalesced writes
        d_out[i] = z;
    }
 
}


// ================
// If the number of taps is small, we can allocate a workspace for the complex-valued inputs
// and then use that workspace to prevent repeated global reads of the same element

// TODO: complete

extern "C" __global__
void filter_smtaps_sminput(
    const complex<float> *d_x, const int len,
    const float *d_taps, const int tapslen,
    const int outputPerBlk, const int workspaceSize,
    complex<float> *d_out, int outlen)
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
    complex<float> z; // stack-var for each thread
    for (int t = threadIdx.x; t < outputPerBlk; t = t + blockDim.x)
    {
        z = 0; // reset before the output

        i = blockDim.x * outputPerBlk + t; // This is the output index

        // Exit if we hit the end
        if (i >= outlen)
            break;

        // Otherwise loop over the taps
        for (int j = 0; j < tapslen; j++)
        {
            // accumulate
            z = d_x[i - j] * s_taps[j];
        }

        // Coalesced writes
        d_out[i] = z;
    }
 
}