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

    // You can only read in from the input within bounds
    if (globalIdx < length)
    {
        // We first read into the thread variable
        x = d_x[globalIdx];
        // Write the corresponding threshold marker
        s_markers[threadIdx.x] = x > threshold ? 1 : 0; // 1 if over threshold, 0 if not
        // Then pre-zero all the edges in shared mem
        s_edges[threadIdx.x] = 0; // This is okay, since 0 cannot be a valid edge by definition (cannot see left neighbour)
    }
    
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
*/
extern "C" __global__
void gatherThresholdEdgesResults(
    const int *d_edges, // (NUM_PREV_BLOCKS * edgesMaxPerBlock)
    const int edgesMaxPerBlock,
    const int NUM_PREV_BLKS,
    const int *d_edgeBlockCounts, // (NUM_PREV_BLOCKS)
    int *d_sliceIndices
){
    // allocate shared memory
    extern __shared__ double s[];

    int *s_edges = (int*)s; // (edgesMaxPerBlock) ints

    // Loop over each counter
    int blockCount;
    int idx = 0; // This is only used by the first thread to know where we are in the output
    bool firstEdgeProcessed = false;
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
                for (int j = 0; j < edgesMaxPerBlock; j++)
                {
                    if (!firstEdgeProcessed && s_edges[j] != 0)
                    {
                        // check if the first one is left or right
                        idx = s_edges[j] > 0 ? 0 : 1; // if its a left edge (+ve) then start at 0, otherwise start at 1
                        firstEdgeProcessed = true;
                    }

                    // from then on, just write the absolute value
                    if (s_edges[j] != 0)
                    {
                        d_sliceIndices[idx] = s_edges[j];
                        idx++;
                    }
  
                }
            } // END OF OUTPUT TO GLOBAL USING FIRST THREAD
        }
    }
}