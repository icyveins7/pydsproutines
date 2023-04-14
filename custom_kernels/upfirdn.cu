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
// TODO: complete

// extern "C" __global__
// void upfirdn_sm(
//     const complex<float> *d_x, const int len,
//     const float *d_taps, const int tapslen,
//     const int up,
//     const int down,
//     complex<float> *d_out,
//     int outlen,
//     float *d_outabs)
// {
//     // allocate shared memory
//     extern __shared__ double s[];
    
//     float *s_taps = (float*)s; // (tapslen) floats
//     /* Tally:  */

//     // load shared memory
//     for (int t = threadIdx.x; t < tapslen; t = t + blockDim.x){
//         s_taps[t] = d_taps[t];
//     }

//     __syncthreads();

//     const int minLookback = blockDim.x * down / up;


    
//     // Define the index that each thread will work on (the output index)
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = i * down; // j will point to the index of the upsampled interrim
//     int k; // we use this to help us point to the original source, depending on the filter tap iteration

//     complex<float> z = 0; // stack-variable for the thread's output

//     // only compute if we are within range
//     if (i < outlen)
//     {
//         // we loop over the filter taps
//         for (int f = 0; f < tapslen; f++)
//         {
//             k = j - f;
//             // again, only accumulate if we are within range, and we are at a non-zero value of the upsample
//             if ((k % up == 0) && (k / up < len) && (k / up >= 0))
//             {
//                 z += s_taps[f] * d_x[k / up];
//             }
//         }

//         // write the output to global memory
//         d_out[i] = z;

//         if (d_outabs != NULL)
//         {
//             d_outabs[i] = abs(z);
//         }
//     }
 
// }
