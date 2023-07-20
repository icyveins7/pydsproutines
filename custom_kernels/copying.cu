#include <cupy/complex.cuh>

// This kernel extracts slices from a 1-D array, specified by
// a Nx2 array sliceBounds, and saves them
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

extern "C" __global__
void copyEqualSlicesToMatrix_32fc(
    const complex<float> *d_x,
    const int xlength, // length of the input, for error checking
    const int *d_xstartIdxs, // each starting input index for the output row, total of outRows
    const int outRows,
    const int outLength,
    complex<float> *d_out // outRows * outLength
){
    int i, row, offset;
    // Grid stride
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int t = tid; t < outRows * outLength; t += gridDim.x * blockDim.x)
    {
        // The row for this thread
        row = t / outLength;
        offset = t % outLength;

        if (row < outRows)
        {
            // The input index for this thread
            i = d_xstartIdxs[row] + offset;
            if (i < xlength) // make sure it's in range
                d_out[row*outLength + offset] = d_x[i];
        }
    }
}