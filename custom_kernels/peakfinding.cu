#include <cooperative_groups.h>

using namespace cooperative_groups;

__device__ int atomicAggInc(int *counter)
{
    auto g = coalesced_threads();
    int warp_res;
    if (g.thread_rank() == 0) // use only the first (leader) thread to increment the counter
        warp_res = atomicAdd(counter, g.size());
    return g.shfl(warp_res, 0) + g.thread_rank(); // return the output position for each (active) thread
}

extern "C" __global__
void findLocalMaxima(
    const float *x,
    const int xlen,
    const float minHeight,
    const int numOutputPerBlk,
    int *peakIndex,
    int *numPeaksFound 
){
    // allocate shared memory
    extern __shared__ double s[];

    float *s_x = (float*)s; // (numOutputPerBlk + 2) floats

    // Define the first input index of this block
    int i0 = numOutputPerBlk * blockIdx.x;

    // Load this batch
    for (int t = threadIdx.x; t < numOutputPerBlk+2; t += blockDim.x)
    {
        int i = i0 - 1 + t; // read starting from -1 of start to +1 of end
        if (i >= 0 && i < xlen)
            s_x[t] = x[i]; 
        else
            s_x[t] = 0;
    }
    __syncthreads();

    // Filter based on local maxima predicate
    for (int t = threadIdx.x; t < numOutputPerBlk; t += blockDim.x)
    {
        int ri = i0 + t; // this is the reference index (where we read from)
        if (ri > xlen) // ignore if we are out of bounds
            break;
        int si = 1 + t; // this is the index it resides in shared mem

        float y = s_x[si];
        // Define our predicate!
        if ((y > minHeight) && (y > s_x[si-1]) && (y > s_x[si+1]))
        {
            peakIndex[atomicAggInc(numPeaksFound)] = ri;
        }
    }
 
}
