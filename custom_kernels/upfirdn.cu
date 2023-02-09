#include <cupy/complex.cuh>
extern "C" __global__
void upfirdn_naive(
    const complex<float> *d_x, const int len,
    const float *d_taps, const int tapslen,
    const int up,
    const int down,
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
            if ((k % up == 0) && (k / up < len))
            {
                z += s_taps[f] * d_x[k / up];
            }
        }

        // write the output to global memory
        d_out[i] = z;
    }
 
}