#include <cupy/complex.cuh>

/*
Currently, these are toy kernels meant to demonstrate masked row kernel launch costs.
See benchmark_maskedkernels.py.
*/



/*
Inputs:
MxN x, MxN y. Mx1 mask.

Based on mask value, each block will either return or perform the multiplication
into the associated output row.

Assumes that number of blocks spawned = number of rows.
*/
extern "C" __global__
void multiplyOnlyMaskedRows(
    const int *mask,
    const complex<float> *x,
    const complex<float> *y,
    complex<float> *out,
    int N,
    int maskValueUsed
){
    // Check mask value for the block
    if (maskValueUsed != mask[blockIdx.x])
        return;

    // Otherwise perform the associated calculation
    for (int t = threadIdx.x; t < N; t += blockDim.x)
    {
        out[blockIdx.x * N + t] = x[blockIdx.x * N + t] * y[blockIdx.x * N + t];
    }
}

/*
Inputs:
MxN x, MxN y0, MxN y1. Mx1 mask.

Based on mask value, each block will perform the multiplication
into the associated output row using either y0 or y1.

Assumes that number of blocks spawned = number of rows.
*/
extern "C" __global__
void multiplyRowsBasedOnMask(
    const int *mask,
    const complex<float> *x,
    const complex<float> *y0,
    const complex<float> *y1,
    complex<float> *out,
    int N
){
    if (mask[blockIdx.x] == 0)
    {
        for (int t = threadIdx.x; t < N; t += blockDim.x)
        {
            out[blockIdx.x * N + t] = x[blockIdx.x * N + t] * y0[blockIdx.x * N + t];
        }
    }
    else
    {
        for (int t = threadIdx.x; t < N; t += blockDim.x)
        {
            out[blockIdx.x * N + t] = x[blockIdx.x * N + t] * y1[blockIdx.x * N + t];
        }
    }
    
}