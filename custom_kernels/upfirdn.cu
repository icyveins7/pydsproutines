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
// Hence spawn as many blocks as there are signals i.e. rows in the input matrix.
// The general idea is to pull in a section of the input, required for a single loop of writing.
// For N threads, each loop will attempt to write N outputs, and reverse calculate the number of inputs
// that will need to be accessed, and place that in a shared memory workspace. This prevents multiple reads
// into global memory. The convention of the workspace length is strictly calculated, and must be replicated
// in the outer calling function to ensure that the required shared memory is available.
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
    // // zero the input workspace (we actually don't need this)
    // for (int t = threadIdx.x; t < inputWorkspaceLength; t = t + blockDim.x){
    //     s_xws[t] = complex<float>(0.0, 0.0);
    // }
    __syncthreads();

    // Point directly to the row for this block for easy access
    const complex<float> *d_row = &d_x[blockIdx.x * len];
    int l0, n0, l, m, n, lws, np0, lp0, globalOffset;
    complex<float> out;

    // Loop over the block until we cover the entire input
    int numLoopsRequired = outlen % blockDim.x == 0 ? outlen / blockDim.x : outlen / blockDim.x + 1;
    for (int i = 0; i < numLoopsRequired; i++)
    {
        // Determine the first output index for this loop
        n0 = i * blockDim.x;
        // Define the global output index for this thread
        n = threadIdx.x + n0;

        // Determine the first input index required
        l0 = (n0 * down - (tapslen-1)) % up == 0 ? (n0 * down - (tapslen-1)) / up : (n0 * down - (tapslen-1)) / up + 1;

        // What was the previous first input index?
        np0 = (i-1) * blockDim.x;
        lp0 = (np0 * down - (tapslen-1)) % up == 0 ? (np0 * down - (tapslen-1)) / up : (np0 * down - (tapslen-1)) / up + 1;

        // Can we move any of the current workspace backwards?
        if (i > 0)
        {
            globalOffset = lp0 + inputWorkspaceLength - l0; // we define this to know how much remainder we need to pull from global mem later
            // we don't make assumptions on the overlap, so we can only use 1 thread for moving
            // otherwise we might end up with some threads writing over the value that another thread is reading
            if (l0 < lp0 + inputWorkspaceLength && threadIdx.x == 0) 
            {
                for (int j = 0; j < globalOffset; j++)
                {
                    s_xws[j] = s_xws[j + l0 - lp0];
                }
            }
            __syncthreads();
        }
        else{ // the first loop does not have an initialised workspace, so we cannot move, just pull all from global mem
            globalOffset = 0; // this tells the next section to read the entire workspace
        }
        
        // Copy the input workspace from global memory, for the remainder
        for (int t = globalOffset + threadIdx.x; t < inputWorkspaceLength; t = t + blockDim.x){
            // Don't read beyond the current row
            if (l0 + t >= 0 && l0 + t < len)
                s_xws[t] = d_row[l0 + t];
            else
                s_xws[t] = 0.0f;
        }
        __syncthreads();

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

                // Accumulate the product if the workspace index is in bounds
                if (lws >= 0 && lws < inputWorkspaceLength)
                    out += s_taps[f] * s_xws[lws];
            }       
        }

        // Write to global output, but only in range
        if (n < outlen)
        {
            d_out[blockIdx.x * outlen + n] = out;
            if (d_outabs != NULL)
                d_outabs[blockIdx.x * outlen + n] = abs(out);
        }
        // You must syncthreads here, or else some warps might start to
        // rewrite the workspace before the entire block is done!
        __syncthreads();
            
    }
 
}
