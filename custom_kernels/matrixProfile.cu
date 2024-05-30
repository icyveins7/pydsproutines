#include <cupy/complex.cuh>

extern "C" __device__ void diagonal_length(const int len, const int k,
                                           const int windowLength) {
  return len - windowLength + 1 - k;
}

extern "C" __global__ void matrix_profile_diagonal(
    const complex<float> *d_x,
    const int len,  // length of d_x
    const int *d_k, // the indices of the diagonals to output
    const int windowLength,
    float *d_diags // output array
) {
  // Lengths do not change..
  // Total length we have to read in from global mem
  int workspaceInputLength = blockDim.x * NUM_PER_THREAD + windowLength - 1;
  // And to output to global mem
  int workspaceOutputLength = blockDim.x * NUM_PER_THREAD;

  // Determine which diagonal this block will work on

  // Where we start reading from global mem for this block
  // This is now 0-offsetted
  int i0 = workspaceOutputLength * blockIdx.x;
  // Where to start writing to global mem for this block
  int o0 = workspaceOutputLength * blockIdx.x;
}
