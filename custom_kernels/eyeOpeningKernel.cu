#include <cupy/complex.cuh>

// each block does one signal in the batch
extern "C" __global__
void getEyeOpening_batch(
    const float *d_abs_batch,
    const int xlength,
    const int osr,
    const complex<float> *d_x_batch,
    complex<float> *d_x_eo_batch)
{
    // allocate shared memory
    extern __shared__ double s[];
    
    float *s_ws = (float*)s; // (osr * blockDim.x) floats
    /* Tally:  */

    // zero shared memory
    for (int i = 0; i < osr; i++)
    {
        s_ws[i * blockDim.x + threadIdx.x] = 0;
    }
    
    __syncthreads();
    
    // load in the abs values and write to the correct eye opening index
    float *d_absx = (float*)&d_abs_batch[xlength * blockIdx.x]; // point to the signal this block will work on
    int j;
    for (int i = threadIdx.x; i < xlength; i += blockDim.x)
    {
        j = i % osr; // define resample index
        s_ws[j * blockDim.x + threadIdx.x] += d_absx[i]; // so we are writing in (osr rows) * (blockDim cols)
    }
    
    __syncthreads();
    
    // now we can sum across each row, which is sequentially addressed
    for (int i = 0; i < osr; i++){
        for (unsigned int s = blockDim.x/2; s > 0; s >>= 1)
        {
            // parallel reduction
            if (threadIdx.x < s){ // if less than half the current 'size'
                s_ws[i * blockDim.x + threadIdx.x] += s_ws[i * blockDim.x + threadIdx.x + s];
            }
            
            __syncthreads();
        }
        // technically can add the warp reduce, but leaving it out for now.....
    }
    
    // now the first index in each row contains the accumulated value, get the max value
    float *s_eo_max = &s_ws[1];
    int *s_eo_argmax = (int*)&s_ws[2]; // we can reuse these 2 in the first row
    
    if (threadIdx.x == 0){ // use first thread
        float eo_max = s_ws[0 * blockDim.x];
        int eo_argmax = 0;
        for (int i = 1; i < osr; i++)
        {
            if (s_ws[i * blockDim.x] > eo_max)
            {
                eo_argmax = i;
                eo_max = s_ws[i * blockDim.x];
            }
        }
        
        *s_eo_argmax = eo_argmax;
        *s_eo_max = eo_max;
        
    }
    __syncthreads();
    
    // now the entire block has access to the max, and the argmax (which is the important one)
    // so we use it to copy the particular eye opening index over
    const complex<float> *d_x = &d_x_batch[blockIdx.x * xlength];
    complex<float> *d_x_eo = &d_x_eo_batch[blockIdx.x * (xlength / osr)]; // DO NOT REMOVE THE CURLY BRACKETS (). IT IS IMPORTANT TO ENSURE THE CORRECT INDEXING.
    int e = *s_eo_argmax;
    for (int i = threadIdx.x; i < xlength / osr; i += blockDim.x)
    {
        d_x_eo[i] = d_x[i * osr + e];
    }
    
}