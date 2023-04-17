#include <cupy/complex.cuh>

// Note that this kernel does not store any of the input array into shared memory.
// It is expected that sufficient blocks are spawned to cover the length of the input.
// See the other version for a one block-one input kernel.
extern "C" __global__
void upfirdn_naive(
    const complex<float> *d_x, const int len,
    const float *d_taps, const int tapslen,
    const int up,
    const int down,
    complex<float> *d_out,
    int outlen,
    float *d_outabs)
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
    
    // Define the index that each thread will work on (the output index)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = i * down; // j will point to the index of the upsampled interrim
    int k; // we use this to help us point to the original source, depending on the filter tap iteration

    complex<float> z = 0; // stack-variable for the thread's output

    // only compute if we are within range
    if (i < outlen)
    {
        // we loop over the filter taps
        for (int f = 0; f < tapslen; f++)
        {
            k = j - f;
            // again, only accumulate if we are within range, and we are at a non-zero value of the upsample
            if ((k % up == 0) && (k / up < len) && (k / up >= 0))
            {
                z += s_taps[f] * d_x[k / up];
            }
        }

        // write the output to global memory
        d_out[i] = z;

        if (d_outabs != NULL)
        {
            d_outabs[i] = abs(z);
        }
    }
 
}

// This kernel attempts to compute upfirdn for one signal in each block.
// Hence spawn as many blocks as there are signals.
// TODO: TEST

extern "C" __global__
void upfirdn_sm(
    const complex<float> *d_x, const int len,
    const float *d_taps, const int tapslen,
    const int up,
    const int down,
    complex<float> *d_out,
    int outlen,
    float *d_outabs)
{
    // Calculate the required input length for the workspace, including filter lookback
    const int interrimLength = ((blockDim.x-1) * down + tapslen);
    const int inputWorkspaceLength = interrimLength % up == 0 ? interrimLength / up : interrimLength / up + 1;

    // allocate shared memory
    extern __shared__ double s[];
    
    float *s_taps = (float*)s; // (tapslen) floats
    complex<float> *s_xws = (complex<float>*)&s_taps[tapslen]; // (inputWorkspaceLength) complex floats

    // load taps
    for (int t = threadIdx.x; t < tapslen; t = t + blockDim.x){
        s_taps[t] = d_taps[t];
    }
    // zero the input workspace
    for (int t = threadIdx.x; t < inputWorkspaceLength; t = t + blockDim.x){
        s_xws[t] = 0;
    }
    __syncthreads();

    // Loop over the block until we cover the entire input
    const complex<float> *d_row = &d_x[blockIdx.x * len];
    int l0, n0, l, m, n, lws;
    complex<float> out;
    int numLoopsRequired = len % blockDim.x == 0 ? len / blockDim.x : len / blockDim.x + 1;
    for (int i = 0; i < numLoopsRequired; i++)
    {
        // Determine the first output index
        n0 = i * blockDim.x;

        // Determine the first input index required
        l0 = (n0 * down - (tapslen-1)) % up == 0 ? (n0 * down - (tapslen-1)) / up : (n0 * down - (tapslen-1)) / up + 1;

        // Copy the input workspace
        for (int t = threadIdx.x; t < inputWorkspaceLength; t = t + blockDim.x){
            // Note that for the first loop, this may be negative, so don't read out of range
            if (l0 + t >= 0)
                s_xws[t] = d_row[l0 + t];
        }
        __syncthreads();

        // Define the global output index for this thread
        n = threadIdx.x + n0;

        // Perform the accumulation
        out = 0;
        for (int f = 0; f < tapslen; f++)
        {
            // What is the interrim index?
            m = n * down - f;

            // Does this correspond to an input index?
            if (m % up == 0)
            {
                // Then what is the global input index?
                l = m / up;
                // What is its associated workspace index?
                lws = l - l0;

                // Accumulate the product
                out += s_taps[f] * s_xws[lws];
            }       
        }

        // Write to global output
        d_out[blockIdx.x * len + n] = out;
        if (d_outabs != NULL)
            d_outabs[blockIdx.x * len + n] = abs(out);
    }
 
}
