#include <cupy/complex.cuh>

/*
We design a kernel to multiply arbitrary slices from an input 1-D array with
indexed arrays from a second input 2-D matrix.

Long input array x is sliced arbitrarily at different indices:
A-B, C-D, E-F, ... where each slice is no longer than length N.

Second input array y is a 2D matrix with individual rows presenting unique
inputs.

The goal is to multiply each slice with a particular row of the matrix. This is specified
with additional index arrays; in general this can be accomplished by iterating over
each slice and multiplying by the appropriate row, at the appropriate length.

However, this kernel attempts to exploit optimistic cases where
number of rows of y <<< number of slices from x.
This means that there is a high probability for the row to be reused from slice to slice,
preventing additional global memory reads.

Each block will allocate shared memory to hold one row, and will swap it out if there is a change
in the row index for the current slice.
*/
extern "C" __global__
void multiplySlicesWithIndexedRowsOptimistic(
    const complex<float> *d_x,
    const int xlength,
    const complex<float> *d_rows,
    const int rowLength,
    const int numRows,
    const int *d_sliceStarts, // length numSlices
    const int *d_sliceLengths, // length numSlices
    const int numSlices,
    const int *d_rowIdxs, // length numSlices
    complex<float> *d_out, // numSlices * rowLength
    int outlength)
{
    // allocate shared memory
    extern __shared__ double s[];

    complex<float> *s_row = (complex<float>*)s; // (rowLength) complex floats

    // On initialization, the loaded index is set to an invalid value
    int loadedRow = -1;
    int requiredRow;

    // Allocate stack variables to hold the current slice indices
    int sliceStart, sliceLength;

    // Iterate over the slices
    for (int i = blockIdx.x; i < numSlices; i += gridDim.x) // each block computes 1 slice at a time
    {
        // First we read the required row index
        requiredRow = d_rowIdxs[i];

        // Then we update shared memory if it's required
        if (requiredRow != loadedRow)
        {
            for (int t = threadIdx.x; t < rowLength; t = t + blockDim.x)
            {
                s_row[t] = d_rows[requiredRow * rowLength + t];
            }

            // Wait for it to be fully loaded
            __syncthreads();
        }

        // Then we perform the multiplies
        sliceStart = d_sliceStarts[i];
        sliceLength = d_sliceLengths[i];

        // Write the output to global mem
        for (int t = threadIdx.x; t < sliceLength; t += blockDim.x)
        {
            if (sliceStart + t >= 0 && sliceStart + t < xlength)
                d_out[i*outlength + t] = s_row[t] * d_x[sliceStart + t];    
        }
            
        
    }

 
}


/*
Here we design a sliding template multiply, with an additional sliding norm calculation.
This is usually used in the xcorr step.

We have two inputs:
1) x: A short (can be contained in shared mem) template array, length xlen
2) y: Another arbitrarily long input array to slide against

This is the MOST OPTIMISTIC method for the copies;
we assume that within the shared memory we can fit both
1) x itself
2) a large section of y

This allows us to slide against multiple xlen windows within 1 block,
escaping a lot of repeated global memory reads.

This is especially important when the template itself is very short;
as the template gets longer this matters less and less when compared to the next 
step in the xcorr, which is the FFT step.

*/
extern "C" __global__ 
void slidingMultiply(
    const complex<float> *x, // the template
    const int xlen,
    const complex<float> *y, // the searched array
    const int ylen,
    const int startIdx, // the start index of the searched array to begin the sliding
    const int idxlen, // the total number of slides
    complex<float> *z, // the output array, which has dimensions (idxlen) rows * (xlen) columns
    float *ynormSq, // the norms of the slices of y, may be left as NULL if undesired, dimensions
    int numSlidesPerBlk // this defines the number of slides to compute per block, and hence determines the workspace (which the caller must calculate correctly)
){
    // allocate shared memory
    extern __shared__ double s[];

    complex<float> *s_x = (complex<float>*)s; // (xlen) complex floats
    complex<float> *s_ysection = (complex<float>*)&s_x[xlen]; // (numSlidesPerBlk + xlen - 1) complex floats
    float *s_ynormSq = (float*)&s_ysection[numSlidesPerBlk + xlen - 1]; // (numSlidesPerBlk) floats

    // Load shared mem x and y
    for (int t = threadIdx.x; t < xlen; t += blockDim.x)
        s_x[t] = x[t];

    int ysectionSize = numSlidesPerBlk + xlen - 1;
    int ysectionOffset = blockIdx.x * numSlidesPerBlk;
    for (int t = threadIdx.x; t < ysectionSize; t += blockDim.x)
        s_ysection[t] = y[ysectionOffset + t];

    // Zero the shared mem norm squared if output is desired
    if (ynormSq != NULL)
    {
        for (int t = threadIdx.x; t < numSlidesPerBlk; t += blockDim.x)
            s_ynormSq[t] = 0.0f;
    }

    // Begin the sliding multiplies
    for (int i = 0; i < numSlidesPerBlk; i++)
    {
        for (int t = threadIdx.x;)
    }
    for (int t = threadIdx.x; t < numSlidesPerBlk; t += blockDim.x)
    {
        for (int i = 0; i < )

        // TODO
    }



}