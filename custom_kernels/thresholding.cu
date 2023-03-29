#include <cupy/complex.cuh>

/*
This kernel aids in burst detection via immediate labelling of threshold edges.
We can first search for all samples which exceed a threshold (example, labelled 1),
and then look for left-only or right-only edges, example:

...1111110
        |
        \-> right edge
01111111...
 |
 \-> left edge

We ignore impulsive noise with both edges like this:
010 -> ignored

In order to process the entire array block-wise, we must ensure that any sample which is
being labelled has access to its neighbouring samples. This means that for N samples read in a block,
we can only write N-2 samples. Inevitably, this means that the 'starting sample' to read for each block
is actually i*(N-2).

Implementation-wise, this kernel is designed to process 1 sample for each thread within a block.
There is no need to loop as we will assume that we will spawn just enough blocks required to cover the input array.
*/
extern "C" __global__
void thresholdEdges(
    const float *d_x,
    const float threshold,
    const int length,
    int *d_edges, // (edgesMaxPerBlock * gridDim.x)
    const int edgesMaxPerBlock,
    int *d_edgeBlockCounts // (gridDim.x)
){
    // allocate shared memory
    extern __shared__ double s[];

    int *s_edges = (int*)s; // blockDim ints
    char *s_markers = (char*)&s_edges[blockDim.x]; // blockDim chars

    // Load the block
    float x;
    int startIdx = (blockDim.x - 2) * blockIdx.x;
    // Define the index that this thread represents in the global array
    int globalIdx = threadIdx.x + startIdx;

    // You can only process the input within bounds
    if (globalIdx < length)
    {
        // We first read into the thread variable
        x = d_x[globalIdx];
        // Write the corresponding threshold marker
        s_markers[threadIdx.x] = x > threshold ? 1 : 0; // 1 if over threshold, 0 if not
        // Then pre-zero all the edges in shared mem
        s_edges[threadIdx.x] = 0; // This is okay, since 0 cannot be a valid edge by definition (cannot see left neighbour)

        // Wait for shared mem
        __syncthreads();

        // === FINDING EDGES ===
        // Ignore the first and last sample
        if (threadIdx.x !=0 && threadIdx.x != blockDim.x-1)
        {
            // First check if the sample passes the threshold
            if (s_markers[threadIdx.x] == 1)
            {
                // Is it a left edge?
                if (s_markers[threadIdx.x-1] == 0 && s_markers[threadIdx.x+1] == 1)
                {
                    s_edges[threadIdx.x] = globalIdx; // write as a 'left edge'
                }
                // Or a right edge?
                else if (s_markers[threadIdx.x-1] == 1 && s_markers[threadIdx.x+1] == 0)
                {
                    s_edges[threadIdx.x] = -globalIdx; // write as a 'right edge' using negative sign
                }
            }
        }
        // Wait for sync
        __syncthreads();

        // We repurpose the markers shared mem here to use 1 integer space, since its no longer needed
        // (generally markers allocation is more than this, requirement is blockDim > 4 threads)
        int *count = (int*)&s_markers[0];

        // Use first thread to push to the front and count
        if (threadIdx.x == 0)
        {
            *count = 0;
            for (int i = 0; i < blockDim.x; i++)
            {
                if (s_edges[i] != 0)
                {
                    s_edges[*count] = s_edges[i];
                    *count = *count + 1;
                    // Re-zero the one we read
                    s_edges[i] = 0;
                }
            }
        }
        // Wait for sync
        __syncthreads();

        // Then push the data back to global memory, up to the count or the max, whichever is smaller
        if (threadIdx.x < *count && threadIdx.x < edgesMaxPerBlock)
        {
            d_edges[blockIdx.x * edgesMaxPerBlock + threadIdx.x] = s_edges[threadIdx.x];
        }
        if (threadIdx.x == 0)
            d_edgeBlockCounts[blockIdx.x] = *count; // write the actual count, for back-checking
    
    }
}

/*
The above kernel outputs results that usually contain many zeroes.
0000x000... processed by block 0 previously
0x0000x0... processed by block 1 previously
and so on..

We also output a counter for each of the blocks in the above kernel
1: from block 0
2: from block 1
0: from block 2 etc.

The goal is to gather all the non-zeros to the front, in one array.
This becomes the final indices. The only safe way to do this is to move all this in
one single block.

Possible scenarios for output allocation:
1) Starts with a right edge, ends with a left edge
    -X .... Y
    Total number of edges is even.
    Required allocation should be +2 of the total length,
    to accomodate 'unknowns' for the first and last slices.

2) Starts with a left edge, ends with a right edge
    X ..... -Y
    Total number of edges is even.
    Required allocation should be exactly the total length.

3) Starts with a left edge, ends with a left edge
    X .... Y
    Total number of edges is odd.
    Required allocation should be exactly +1 of the total length,
    to accomodate 'unknown' for the last slice.

4) Starts with a right edge, ends with a right edge
    -X .... -Y
    Total number of edges is odd.
    Required allocation should be exactly +1 of the total length,
    to accomodate 'unknown' for the first slice.

Note that the first edge may appear in the N'th row, so it is not possible to know beforehand
and output the left/right-ness of the first edge from the previous kernel.
Therefore, the best case should be even->+2 length, odd->+1 length.
*/
extern "C" __global__
void gatherThresholdEdgesResults(
    const int *d_edges, // (NUM_PREV_BLOCKS * edgesMaxPerBlock)
    const int edgesMaxPerBlock,
    const int NUM_PREV_BLKS,
    const int *d_edgeBlockCounts, // (NUM_PREV_BLOCKS)
    const int minimumWidth, // if not required, set to 0
    const int maximumWidth, // if not required, set to int32 max i.e. 2147483647
    int *d_sliceIndices,
    int *d_totalCount
){
    // allocate shared memory
    extern __shared__ double s[];

    int *s_edges = (int*)s; // (edgesMaxPerBlock) ints
    
    // Loop over each counter
    int blockCount;
    int idx = 0; // This is only used by the first thread to know where we are in the output

    int left = 0, right = 0;
    for (int i = 0; i < NUM_PREV_BLKS; i++)
    {
        blockCount = d_edgeBlockCounts[i];
        // Only read from the edges array if necessary
        if (blockCount > 0)
        {
            if (threadIdx.x < edgesMaxPerBlock)
                s_edges[threadIdx.x] = d_edges[i*edgesMaxPerBlock + threadIdx.x];

            __syncthreads(); // Wait for this row to be moved in

            // Then use first thread to push into the global output
            if (threadIdx.x == 0)
            {
                // We loop over each edge, but only write once we've found a pair
                for (int j = 0; j < edgesMaxPerBlock; j++)
                {
                    // if this is a left edge, just cache it for now
                    if (s_edges[j] > 0)
                        left = s_edges[j];

                    // if its a right edge,
                    else if (s_edges[j] < 0)
                    {
                        right = abs(s_edges[j]); // take the abs
                        // check whether it satisfies the limits
                        if (right - left >= minimumWidth && right - left <= maximumWidth)
                        {
                            // then we write to global
                            d_sliceIndices[idx] = left;
                            idx++;
                            d_sliceIndices[idx] = right;
                            idx++;
                            // reset
                            left = 0;
                            right = 0;
                        }
                    }
  
                }
            } // END OF OUTPUT TO GLOBAL USING FIRST THREAD
        }
    }

    // At the very end, update the total useful count
    if (threadIdx.x == 0)
        d_totalCount[0] = idx / 2;
}