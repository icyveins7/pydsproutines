#include <cupy/complex.cuh>

/*
Assuming a 4-D array, each block gets the 3-D argmax of the last 3 dimensions.

Different flavours for different input types,
but all returned indices are in uint32.
*/

// Input type: uint32
extern "C" __global__
void multiArgmax3d_uint32(
    const unsigned int *d_x,
    const int numItems,
    const int dim1,
    const int dim2,
    const int dim3, // so d_x is expected to be (numItems * dim1 * dim2 * dim3)
    unsigned int *d_argmax, // output is expected to be (numItems * 3)
    unsigned int *d_max // length of (numItems)
){
    // allocate shared memory
    extern __shared__ double s[];

    unsigned int *s_item = (unsigned int*)s; // (blockDim) unsigned ints
    unsigned int *s_idx = (unsigned int*)&s_item[blockDim.x]; // (blockDim) unsigned ints

    // First we zero both shared mem workspaces
    s_item[threadIdx.x] = 0;
    s_idx[threadIdx.x] = 0;
    // no need to sync here, each thread goes on to its own reads and comparisons first

    // extract the item for this block
    int itemSize = dim1 * dim2 * dim3;
    unsigned int item;
    for (int t = threadIdx.x; t < itemSize; t += blockDim.x)
    {
        item = d_x[itemSize * blockIdx.x + t];
        if (item > s_item[threadIdx.x])
        {
            // replace the value, and write the index
            s_item[threadIdx.x] = item;
            s_idx[threadIdx.x] = t;
        }
    }
    __syncthreads();

    // Parallel reduction maximum
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            // retain the larger value
            if (s_item[threadIdx.x] < s_item[threadIdx.x + s])
            {
                s_item[threadIdx.x] = s_item[threadIdx.x + s];
                s_idx[threadIdx.x] = s_idx[threadIdx.x + s]; // remember to take the index too
            }
        }

        __syncthreads();
    }

    // Now the max value and index is at the zero index in the workspaces
    unsigned int rem = s_idx[0];
    unsigned int out1 = rem / (dim2 * dim3);
    rem = rem % (dim2 * dim3);
    unsigned int out2 = rem / (dim3);
    rem = rem % dim3;
    unsigned int out3 = rem;
    
    // Execute 3-D write with just the first 3 threads
    if (threadIdx.x == 0)
        d_argmax[blockIdx.x * 3 + 0] = out1;
    if (threadIdx.x == 1)
        d_argmax[blockIdx.x * 3 + 1] = out2;
    if (threadIdx.x == 2)
        d_argmax[blockIdx.x * 3 + 2] = out3;
    if (d_max != NULL && threadIdx.x == 3)
        d_max[blockIdx.x] = s_item[0];

}