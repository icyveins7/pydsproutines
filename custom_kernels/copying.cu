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

/*
For sliding windows, each row in the output is simply a fixed index offset from the previous row, e.g.
i  , i+1, i+2, ...
i+2, i+3, i+4, ...
i+4, i+5, i+6, ...
...

In cases like this, we can reduce global reads by accessing a square block;
this requires a single global coalesced read into shared memory, and then multiple global coalesced writes from shared memory.

Common configurations for shared mem dimensions would be:
32(rows) * 128(columns) => 32768 bytes, use 128 threads per block
16(rows) * 256(columns) => 32768 bytes, use 256 threads per block

Note that the shared memory will very likely not be fully used,
unless the increment is equal to the number of columns. 
But overallocating this makes it simpler to deal with, and gives us a way to track the output rectangle.
*/
extern "C" __global__
void copyIncrementalEqualSlicesToMatrix_32fc(
    const complex<float> *d_x,
    const int xlength, // length of the input, for error checking
    const int startIdx, // this is the input starting index for the first row
    const int increment, // this is the jump from one row to the next
    const int outRows,
    const int outLength,
    const int blockRows, const int blockCols, // we need this so that our block can find which portion to copy
    complex<float> *d_out // outRows * outLength
){
    // Allocate shared memory
    extern __shared__ double s[];

    complex<float> *s_ws = (complex<float> *)s; // (blockRows * blockCols)

    // Define the first output row for this block
    int firstOutputRow = blockRows * blockIdx.y;
    // Define the first output column for this block
    int firstOutputCol = blockCols * blockIdx.x;
    // Define the first input index for this block
    int firstInputIndex = firstOutputRow * increment + startIdx + firstOutputCol;
    // Define the last input index for this block (not inclusive)
    int lastInputIndex = firstInputIndex + (blockRows-1) * increment + blockCols;

    // Copy into shared memory
    for (int i = threadIdx.x; i < lastInputIndex-firstInputIndex; i += blockDim.x)
    {
        if (firstInputIndex + i < xlength) // dont read out of bounds
            s_ws[i] = d_x[firstInputIndex + i];
        else
            s_ws[i] = 0;
    }
    __syncthreads();

    // Now we read from shared memory and just write to global
    for (int i = 0; i < blockRows; i++)
    {
        int row = firstOutputRow + i;
        // Ensure writes are in bounds
        if (row < outRows)
        {
            for (int t = threadIdx.x; t < blockCols; t += blockDim.x)
            {
                int col = firstOutputCol + t;
                if (col < outLength) // Ensure writes are in bounds
                    d_out[row*outLength + firstOutputCol + t] = s_ws[t + i*increment];
            }
        }
        
    }
    
}