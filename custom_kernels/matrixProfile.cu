#include <cupy/complex.cuh>

extern "C" __device__ void diagonal_length(const int len, const int k,
                                           const int windowLength) {
  return len - windowLength + 1 - k;
}

extern "C" __global__ void matrix_profile_diagonal(
    const complex<float> *d_x, const float *d_xnormSqs,
    const int len,               // length of d_x
    const int *d_k,              // the indices of the diagonals to output
    const int blocksPerDiagonal, // to help find out which diagonal to compute
    const int windowLength, const int NUM_PER_THREAD, const float minThreshold,
    float *d_diags // output array
) {
  // Lengths do not change..
  // Total length we have to read in from global mem
  int workspaceInputLength = blockDim.x * NUM_PER_THREAD + windowLength - 1;
  // And to output to global mem
  int workspaceOutputLength = blockDim.x * NUM_PER_THREAD;

  // Determine which diagonal this block will work on
  int k = d_k[blockIdx.x / blocksPerDiagonal];

  // Where we start reading from global mem for this block
  // This is now 0-offsetted
  int i0 = workspaceOutputLength * blockIdx.x;
  int j0 =
      workspaceOutputLength * blockIdx.x + k; // the offset slice, needs conj

  // Where to start writing to global mem for this block
  int o0 = workspaceOutputLength * blockIdx.x;

  // allocate shared memory
  extern __shared__ double s[];

  complex<float> *s_pdt = (complex<float> *)s; // (workspaceInputLength) floats
  complex<float> *s_ws =
      (complex<float>
           *)&s_pdt[workspaceInputLength]; // (workspaceOutputLength) floats

  // Read both the front and back slices, and perform the complex mul directly
  // Do it with only 1 store to shared memory
  for (int t = threadIdx.x; t < workspaceInputLength; t += blockDim.x) {
    // Only read if it's in range
    int iidx = t + i0;
    int jidx = t + j0;
    if (iidx >= 0 && iidx < xlen && jidx >= 0 && jidx < xlen)
      s_pdt[t] = d_x[iidx] * d_x[jidx].conj();
    else
      s_pdt[t] = 0;
  }
  __syncthreads();

  // Now perform the averaging, same as our moving window kernels
  // in filter.cu

  // Compute first value
  complex<double> sum =
      0; // We internally calculate as a double to have higher precision
  for (int i = 0; i < sumLength; i++) {
    sum += (complex<double>)s_x[threadIdx.x * NUM_PER_THREAD + i];
  }
  s_ws[threadIdx.x * NUM_PER_THREAD] = (complex<float>)sum;

  // Compute the rest
  for (int j = 1; j < NUM_PER_THREAD; j++) {
    sum -= (complex<double>)s_x[threadIdx.x * NUM_PER_THREAD + j - 1];
    sum +=
        (complex<double>)s_x[threadIdx.x * NUM_PER_THREAD + sumLength + j - 1];
    // We either save the sum or the average
    s_ws[threadIdx.x * NUM_PER_THREAD + j] = (complex<float>)sum;
  }

  // Calculate magnitudes (QF2) and scale by the normSq values in global mem
  // Write these to the front of the input array in shared mem (reuse it)

  // Then now extract valid chains that pass the threshold
  // Write these to global memory in an atomic way (see peakfinding.cu)

  // // Finally write our output coalesced to global
  // for (int t = threadIdx.x; t < workspaceOutputLength; t += blockDim.x) {
  //   if (o0 + t < xlen)
  //     d_out[o0 + t] = abs(s_ws[t]) * abs(s_ws[t]);
  // }
}
