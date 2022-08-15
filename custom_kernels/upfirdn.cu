#include <cupy/complex.cuh>
extern "C" __global__
void upfirdn_sm(
    const complex<float> *d_x, const int len,
    const float *d_taps, const int tapslen,
    const int up,
    const int down,
    const int shm_x_size,
    complex<float> *d_out, int outlen, float *d_outabs)
{
    // allocate shared memory
    extern __shared__ double s[];
    
    float *s_taps = (float*)s; // (tapslen) floats
    complex<float> *s_x = (complex<float>*)&s_taps[tapslen]; // (shm_x_size) complex floats
    /* Tally:  */

    // load shared memory
    for (int t = threadIdx.x; t < tapslen; t = t + blockDim.x){
        s_taps[t] = d_taps[t];
    }
    
    // Define the indices to write to for this block
    int outStart = blockIdx.x * blockDim.x + tapslen / 2;
    int outEnd = min((blockIdx.x + 1) * blockDim.x + tapslen / 2, outlen + tapslen/2);
    
    // calculate the offset for this block
    int blockReadOffset = (outStart * down - tapslen) / up; // TODO: define this
    // note that shm_x_size must this extra front buffer as well
    for (int t = threadIdx.x; t < shm_x_size; t = t + blockDim.x)
    {
        if (t + blockReadOffset >= 0 && t + blockReadOffset < len){ // only read if in range
            s_x[t] = d_x[t + blockReadOffset];
        }
        else{
            s_x[t] = 0;
        }
    }
    __syncthreads();
    
    // Begin computations
    int i0, j;
    complex<float> z = 0; // Stack-variable for each thread
    
    // Make it simple, every thread writes 1 output
    int k = threadIdx.x + outStart;
    if (k < outEnd)
    {
        for (int i = 0; i < tapslen; i++)
        {
            i0 = down * k - i;
            // don't bother reading if its non-zero
            if (i0 % up == 0){
                j = i0 / up; // this is the access into the 'x' array
                j = j - blockReadOffset; // we only copied a section into shared memory, so change the index
                
                if (j < shm_x_size && j >= 0) // cannot read out of bounds
                {
                    z = z + s_taps[i] * s_x[j];
                }
            }
            
        }
        
        // write to global memory, coalesced, and offset half the filter automatically
        d_out[k - tapslen / 2] = z;
        d_outabs[k - tapslen / 2] = abs(z);
    }

 
}