#include <cooperative_groups.h>
#include <cupy/complex.cuh>

using namespace cooperative_groups;

extern "C" __device__ int diagonal_length(const int len, const int k,
                                          const int windowLength) {
  return len - windowLength + 1 - k;
}

__device__ int atomicAggInc(int *counter) {
  auto g = coalesced_threads();
  int warp_res;
  if (g.thread_rank() ==
      0) // use only the first (leader) thread to increment the counter
    warp_res = atomicAdd(counter, g.size());
  return g.shfl(warp_res, 0) +
         g.thread_rank(); // return the output position for each (active) thread
}

/*
This kernel will compute every single output of the matrix (upper triangle).
This can get excessively big.

The way this is organised is the following:
1) The output is allocated to the exact length required for all diagonals.
This is comprised of the longest (k=1) diagonal all the way to the last diagonal
with length 1. As such Gauss' method can be used to calculate the total number
of elements.

2) The blocks are allocated using an externally calculated blocksPerDiagonal.
This allocates a total of numDiagonals * blocksPerDiagonal blocks.
This value should cover the longest (k=1) diagonal's outputs just nicely.
However, the later diagonals will have many 'useless' blocks, which we ignore as
the kernel overhead is probably negligible in the grand scheme of things.

Example:
k=1     All blocks used.
...
k=10    1 block unused.
...
k=100   3 blocks unused.
...
k_final All blocks unused except 1.

3) As such, each block needs to first figure out which diagonal (value of k) it
is working on, and then figure out which output for that diagonal it has been
assigned to (blkOffset).

*/
extern "C" __global__ void matrix_profile_raw(
    const complex<float> *d_x, const float *d_xnormSqs,
    const int len,               // length of d_x
    const int blocksPerDiagonal, // to help find out which diagonal to compute
    const int windowLength, const int NUM_PER_THREAD,
    float *d_out // output array
) {
  // Lengths do not change..
  // Total length we have to read in from global mem
  int workspaceInputLength = blockDim.x * NUM_PER_THREAD + windowLength - 1;
  // And to output to shared mem
  int workspaceOutputLength = blockDim.x * NUM_PER_THREAD;

  // Determine which diagonal this block works on
  int k = blockIdx.x / blocksPerDiagonal + 1; // first one is 1, not 0
  // Length of the diagonal
  int diagLength = diagonal_length(len, k, windowLength);

  // // Early exit if the accessed index is out of bounds
  // if (j0 >= len)
  //   return;

  // Where to start writing to global mem for this block
  // We use gauss's method to compute the offset
  // k = 1   -> N - L + 1 - 1 = N - L      => offset is 0
  // k = 2   -> N - L + 1 - 2 = N - L - 1  => offset is N-L
  // k = 3   -> N - L + 1 - 3              => offset is N-L + N-L-1
  // ....
  // k       -> N - L + 1 - k = diagLength
  // ....
  // k = N-L -> N - L + 1 - (N-L) = 1
  // NOTE: there are a total of N-L diagonals
  // So this is the offset to write to the k'th diagonal,
  // which is the sum of elements from the 1st to the k-1'th diagonal
  int ok0 =
      (len - windowLength + diagonal_length(len, k - 1, windowLength)) *
      (len - windowLength - diagonal_length(len, k - 1, windowLength) + 1) / 2;

  // And this is where (within the diagonal output) the block actually writes
  int blkOffset =
      (blockIdx.x % blocksPerDiagonal) * blockDim.x * NUM_PER_THREAD;
  int okb0 = ok0 + blkOffset;

  // Where we start reading from global mem for this block
  // This is now 0-offsetted
  int i0 = workspaceOutputLength * blkOffset;
  int j0 =
      workspaceOutputLength * blkOffset + k; // the offset slice, needs conj

  // allocate shared memory
  extern __shared__ double s[];

  complex<float> *s_pdt =
      (complex<float> *)s; // (workspaceInputLength) complex floats
  complex<float> *s_ws =
      (complex<float>
           *)&s_pdt[workspaceInputLength]; //  (workspaceOutputLength)
  // complex floats

  // Read both the front and back slices, and perform the complex mul
  // directly
  // Do it with only 1 store to shared memory
  for (int t = threadIdx.x; t < workspaceInputLength; t += blockDim.x) {
    // Only read if it's in range
    int iidx = t + i0;
    int jidx = t + j0;
    if (iidx >= 0 && iidx < len && jidx >= 0 && jidx < len)
      s_pdt[t] = d_x[iidx] * conj(d_x[jidx]);
    else
      s_pdt[t] = 0;
  }
  __syncthreads();

  // Now perform the averaging, same as our moving window kernels
  // in filter.cu

  // Compute first value
  complex<double> sum =
      0; // We internally calculate as a double to have higher precision
  for (int i = 0; i < windowLength; i++) {
    sum += (complex<double>)s_pdt[threadIdx.x * NUM_PER_THREAD + i];
  }
  s_ws[threadIdx.x * NUM_PER_THREAD] = (complex<float>)sum;

  // Compute the rest
  for (int j = 1; j < NUM_PER_THREAD; j++) {
    sum -= (complex<double>)s_pdt[threadIdx.x * NUM_PER_THREAD + j - 1];
    sum += (complex<double>)
        s_pdt[threadIdx.x * NUM_PER_THREAD + windowLength + j - 1];
    // We either save the sum or the average
    s_ws[threadIdx.x * NUM_PER_THREAD + j] = (complex<float>)sum;
  }
  __syncthreads();

  for (int t = threadIdx.x; t < workspaceOutputLength; t += blockDim.x) {
    if (t + blkOffset < diagLength) {
      // d_out[okb0 + t] = threadIdx.x;
      // d_out[okb0 + t] = i0 + t;
      // d_out[okb0 + t] = j0 + t;

      d_out[okb0 + t] =
          norm(s_ws[t]) / (d_xnormSqs[i0 + t] * d_xnormSqs[j0 + t]);
    }
  }
}

/*
Most of the front code for this kernel is copied from the above.
We diverge at the global writes, choosing to filter the output in shared memory
into the chains first.
*/
extern "C" __global__ void matrix_profile_chains(
    const complex<float> *d_x, const float *d_xnormSqs,
    const int len,               // length of d_x
    const int *d_k,              // the indices of the diagonals to output
    const int blocksPerDiagonal, // to help find out which diagonal to compute
    const int windowLength, const int NUM_PER_THREAD, const float minThreshold,
    int *numChains, // scalar counter to atomically increment d_chains
    int *d_chains   // output array, atomically added 3-tuples (x, y = x+k, len)
) {
  // Lengths do not change..
  // Total length we have to read in from global mem
  int workspaceInputLength = blockDim.x * NUM_PER_THREAD + windowLength - 1;
  // And to output to shared mem
  int workspaceOutputLength = blockDim.x * NUM_PER_THREAD;

  // Determine which diagonal this block works on
  int k = blockIdx.x / blocksPerDiagonal + 1; // first one is 1, not 0
  // Length of the diagonal
  int diagLength = diagonal_length(len, k, windowLength);

  // // Early exit if the accessed index is out of bounds
  // if (j0 >= len)
  //   return;

  // Where to start writing to global mem for this block
  // We use gauss's method to compute the offset
  // k = 1   -> N - L + 1 - 1 = N - L      => offset is 0
  // k = 2   -> N - L + 1 - 2 = N - L - 1  => offset is N-L
  // k = 3   -> N - L + 1 - 3              => offset is N-L + N-L-1
  // ....
  // k       -> N - L + 1 - k = diagLength
  // ....
  // k = N-L -> N - L + 1 - (N-L) = 1
  // NOTE: there are a total of N-L diagonals
  // So this is the offset to write to the k'th diagonal,
  // which is the sum of elements from the 1st to the k-1'th diagonal
  int ok0 =
      (len - windowLength + diagonal_length(len, k - 1, windowLength)) *
      (len - windowLength - diagonal_length(len, k - 1, windowLength) + 1) / 2;

  // And this is where (within the diagonal output) the block actually writes
  int blkOffset =
      (blockIdx.x % blocksPerDiagonal) * blockDim.x * NUM_PER_THREAD;
  int okb0 = ok0 + blkOffset;

  // Where we start reading from global mem for this block
  // This is now 0-offsetted
  int i0 = workspaceOutputLength * blkOffset;
  int j0 =
      workspaceOutputLength * blkOffset + k; // the offset slice, needs conj

  // allocate shared memory
  extern __shared__ double s[];

  complex<float> *s_pdt =
      (complex<float> *)s; // (workspaceInputLength) complex floats
  complex<float> *s_ws =
      (complex<float>
           *)&s_pdt[workspaceInputLength]; //  (workspaceOutputLength)
  // complex floats

  // Read both the front and back slices, and perform the complex mul
  // directly
  // Do it with only 1 store to shared memory
  for (int t = threadIdx.x; t < workspaceInputLength; t += blockDim.x) {
    // Only read if it's in range
    int iidx = t + i0;
    int jidx = t + j0;
    if (iidx >= 0 && iidx < len && jidx >= 0 && jidx < len)
      s_pdt[t] = d_x[iidx] * conj(d_x[jidx]);
    else
      s_pdt[t] = 0;
  }
  __syncthreads();

  // Now perform the averaging, same as our moving window kernels
  // in filter.cu

  // Compute first value
  complex<double> sum =
      0; // We internally calculate as a double to have higher precision
  for (int i = 0; i < windowLength; i++) {
    sum += (complex<double>)s_pdt[threadIdx.x * NUM_PER_THREAD + i];
  }
  s_ws[threadIdx.x * NUM_PER_THREAD] = (complex<float>)sum;

  // Compute the rest
  for (int j = 1; j < NUM_PER_THREAD; j++) {
    sum -= (complex<double>)s_pdt[threadIdx.x * NUM_PER_THREAD + j - 1];
    sum += (complex<double>)
        s_pdt[threadIdx.x * NUM_PER_THREAD + windowLength + j - 1];
    // We either save the sum or the average
    s_ws[threadIdx.x * NUM_PER_THREAD + j] = (complex<float>)sum;
  }
  __syncthreads();

  // ========= Here is where we diverge from the raw output kernel

  // Calculate magnitudes (QF2) and scale by the normSq values in global mem
  // Write these to the front of the input array in shared mem (reuse it)
  float *s_qf2 = (float *)s_ws;
  for (int t = threadIdx.x; t < workspaceOutputLength; t += blockDim.x) {
    if (t + blkOffset < diagLength)
      s_qf2[t] = norm(s_ws[t]) / (d_xnormSqs[i0 + t] * d_xnormSqs[j0 + t]);
    else
      s_qf2[t] = 0;
  }
  __syncthreads();

  // Define the current chain
  int chain[3] = {-1, -1, -1};
  // Iterate over the QF2 values in shared mem
  for (int j = 0; j < NUM_PER_THREAD; j++) {
    int si = threadIdx.x * NUM_PER_THREAD + j;
    // Don't evaluate anything that's out of bounds
    if (si >= workspaceOutputLength)
      break; // This shouldn't even occur anyway?

    // Current element passes
    if (s_qf2[si] > minThreshold) {
      // Chain already exists, then append
      if (chain[0] != -1)
        chain[2] += 1;

      // Otherwise start a new chain
      else {
        chain[0] = i0 + si;
        chain[1] = j0 + si;
        chain[2] = 1;
      }
    }
    // Current element doesn't pass
    else {
      // If chain exists, write it, and clear it back to -1s
      if (chain[0] != -1) {
        int wIdx = atomicAggInc(numChains);
        d_chains[wIdx * 3 + 0] = chain[0];
        d_chains[wIdx * 3 + 1] = chain[1];
        d_chains[wIdx * 3 + 2] = chain[2];

        chain[0] = -1;
        chain[1] = -1;
        chain[2] = -1;
      }
      // Otherwise don't need to do anything
    }
  }
}
