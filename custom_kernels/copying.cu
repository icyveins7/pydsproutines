#include <cupy/complex.cuh>

// This kernel extracts slices from a 1-D array, specified by
// two arrays 'sliceStarts' and 'sliceEnds', and saves them
// into a 2-d array, with each one occupying a row.
// The output 2-d array is expected to have enough columns to contain all lengths.
// This avoids a pythonic loop, as each block tackles a row i.e. a slice.
extern "C" __global__
void copySlicesToMatrix_32fc(
    const complex<float> *d_x,
    const int xlength,
    const int *d_sliceBounds, // numSlices x 2
    const int numSlices,
    const int rowLength,
    complex<float> *d_out // numSlices * rowLength
){
    // simple checks
    if (blockIdx.x >= numSlices)
        return;

    // define the slice values for this block
    const int sliceStart = d_sliceBounds[blockIdx.x*2+0];
    const int sliceEnd = d_sliceBounds[blockIdx.x*2+1];

    // define read pointer for the block
    const complex<float> *x = &d_x[sliceStart];

    // define write pointer for the block
    complex<float> *out = &d_out[rowLength * blockIdx.x];

    // perform the copy
    for (int t = threadIdx.x; t < sliceEnd-sliceStart; t += blockDim.x)
    {
        // don't read/write out of bounds
        if (t + sliceStart >= 0 && t + sliceStart < xlength && t < rowLength)
            out[t] = x[t];
    }
}