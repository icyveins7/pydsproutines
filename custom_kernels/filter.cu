#include <cupy/complex.cuh>

// This kernel only loads the filter taps into shared memory, without an extra
// workspace This is useful if the number of filter taps is very long, and hence
// leaves less than 2*length of complex workspace for the data to sit in.
// Relevant only for real-only filter taps.
// In order to not waste warps, it is recommended to set outputPerBlk to a
// multiple of blockDim.
extern "C" __global__ void
filter_smtaps(const complex<float> *d_x, const int len, const float *d_taps,
              const int tapslen, const int outputPerBlk, complex<float> *d_out,
              int outlen, const complex<float> *d_delay, const int delaylen,
              const int dsr, const int dsPhase) {
  // allocate shared memory
  extern __shared__ double s[];

  float *s_taps = (float *)s; // (tapslen) floats
  /* Tally:  */

  // load shared memory
  for (int t = threadIdx.x; t < tapslen; t = t + blockDim.x) {
    s_taps[t] = d_taps[t];
  }

  __syncthreads();

  // Begin computations
  int i;            // output index
  int k;            // reference index (not equal to output if downsample is >1)
  complex<float> z; // stack-var for each thread
  for (int t = threadIdx.x; t < outputPerBlk; t = t + blockDim.x) {
    z = 0; // reset before the output

    i = blockIdx.x * outputPerBlk + t; // This is the thread's output index
    k = i * dsr + dsPhase;             // This is the reference index

    // Exit if we hit the end
    if (i >= outlen)
      break;

    // Otherwise loop over the taps
    for (int j = 0; j < tapslen; j++) {
      int xIdx = k - j;

      // accumulate
      if (xIdx >= 0 && xIdx < len)
        z = z + d_x[xIdx] * s_taps[j]; // this uses the input data
      else if (delaylen + xIdx >= 0 &&
               d_delay != NULL) // d_delay must be supplied for this to work
        z = z +
            d_delay[delaylen + xIdx] * s_taps[j]; // this uses the delay data
                                                  // (from previous invocations)
    }

    // Coalesced writes
    d_out[i] = z;
  }
}

// Literally identical to above but for non-complex inputs
extern "C" __global__ void filter_smtaps_sminput_real(
    const float *d_x, const int len, const float *d_taps, const int tapslen,
    const int outputPerBlk,
    const int
        workspaceSize, // this must correspond to outputPerBlk + tapslen - 1
    float *d_out, int outlen) {
  // allocate shared memory
  extern __shared__ double s[];

  float *s_taps = (float *)s;              // (tapslen) floats
  float *s_ws = (float *)&s_taps[tapslen]; // workspaceSize
  /* Tally:  */

  // load shared memory taps
  for (int t = threadIdx.x; t < tapslen; t = t + blockDim.x) {
    s_taps[t] = d_taps[t];
  }
  // load the shared memory workspace
  int i0 = blockIdx.x * outputPerBlk; // this is the first output index
  int workspaceStart =
      i0 - tapslen + 1; // this is the first index that is required
  // int workspaceEnd   = i0 + outputPerBlk; // this is the last index that is
  // required (non-inclusive)
  int i;
  for (int t = threadIdx.x; t < workspaceSize; t = t + blockDim.x) {
    i = workspaceStart + t;   // this is the input source index to copy
    if (i < 0 || i >= outlen) // set to 0 if its out of range
      s_ws[t] = 0;
    else
      s_ws[t] = d_x[i];
  }

  __syncthreads();

  // Begin computations
  float z; // stack-var for each thread
  int wsi;
  for (int t = threadIdx.x; t < outputPerBlk; t = t + blockDim.x) {
    z = 0; // reset before the output

    i = blockIdx.x * outputPerBlk + t; // This is the output index
    wsi = tapslen - 1 +
          t; // this is the 'equivalent' source index from shared memory

    // Exit if we hit the end
    if (i >= outlen)
      break;

    // Otherwise loop over the taps and the shared mem workspace
    for (int j = 0; j < tapslen; j++) {
      // accumulate
      z = z + s_ws[wsi - j] * s_taps[j];
    }

    // Coalesced writes
    d_out[i] = z;
  }
}

// ================
// If the number of taps is small, we can allocate a workspace for the
// complex-valued inputs and then use that workspace to prevent repeated global
// reads of the same element
extern "C" __global__ void filter_smtaps_sminput(
    const complex<float> *d_x, const int len, const float *d_taps,
    const int tapslen, const int outputPerBlk,
    const int
        workspaceSize, // this must correspond to outputPerBlk + tapslen - 1
    complex<float> *d_out, int outlen) {
  // allocate shared memory
  extern __shared__ double s[];

  float *s_taps = (float *)s;                                // (tapslen) floats
  complex<float> *s_ws = (complex<float> *)&s_taps[tapslen]; // workspaceSize
  /* Tally:  */

  // load shared memory taps
  for (int t = threadIdx.x; t < tapslen; t = t + blockDim.x) {
    s_taps[t] = d_taps[t];
  }
  // load the shared memory workspace
  int i0 = blockIdx.x * outputPerBlk; // this is the first output index
  int workspaceStart =
      i0 - tapslen + 1; // this is the first index that is required
  // int workspaceEnd   = i0 + outputPerBlk; // this is the last index that is
  // required (non-inclusive)
  int i;
  for (int t = threadIdx.x; t < workspaceSize; t = t + blockDim.x) {
    i = workspaceStart + t;   // this is the input source index to copy
    if (i < 0 || i >= outlen) // set to 0 if its out of range
      s_ws[t] = 0;
    else
      s_ws[t] = d_x[i];
  }

  __syncthreads();

  // Begin computations
  complex<float> z; // stack-var for each thread
  int wsi;
  for (int t = threadIdx.x; t < outputPerBlk; t = t + blockDim.x) {
    z = 0; // reset before the output

    i = blockIdx.x * outputPerBlk + t; // This is the output index
    wsi = tapslen - 1 +
          t; // this is the 'equivalent' source index from shared memory

    // Exit if we hit the end
    if (i >= outlen)
      break;

    // Otherwise loop over the taps and the shared mem workspace
    for (int j = 0; j < tapslen; j++) {
      // accumulate
      z = z + s_ws[wsi - j] * s_taps[j];
    }

    // Coalesced writes
    d_out[i] = z;
  }
}

/*
Allocate a grid with (x,y) blocks such that gridDim.y == numRows.
Each thread computes 1 output.

Naive implementation:
Each thread sums over its own window, and thread windows overlap.
E.g.
Thread 0: 0 -> N-1
Thread 1: 1 -> N
Thread 2: 2 -> N+1
...
Each block will have N threads, which works on N outputs of a particular row.
*/
extern "C" __global__ void
multiMovingAverage(const float *d_x, const int numRows,
                   const int numCols,   // same dimensions for d_x and d_out
                   const int avgLength, // moving average window size,
                   float *d_out) {
  // allocate shared memory
  extern __shared__ double s[];

  float *s_x = (float *)s; // (tapslen) floats
  /* Tally:  */

  // Calculate this block's ROW offset
  int blockOffset = blockIdx.y * numCols;
  // Calculate shared memory usage (each thread computes 1 output)
  int sharedMemSize = blockDim.x + avgLength - 1;
  // Determine the first index of this block's input row to start copying
  int i0 = blockIdx.x * blockDim.x - avgLength + 1; // this may be negative!

  // Copy the shared memory, setting 0s if it references a negative index
  for (int t = threadIdx.x; t < sharedMemSize; t += blockDim.x) {
    // Evaluate the column index for this block to read from
    int idx = t + i0;
    if (idx >= 0 && idx < numCols)
      s_x[t] = d_x[blockOffset + idx];
    else
      s_x[t] = 0;
  }

  __syncthreads();

  // Now compute
  for (int t = threadIdx.x; t < blockDim.x; t += blockDim.x) {
    // No writing out of bounds
    if (blockIdx.x * blockDim.x + t < numCols) {
      double sum = 0.0;

      for (int i = 0; i < avgLength; i++) {
        sum += s_x[t + i];
      }

      d_out[blockOffset + blockIdx.x * blockDim.x + t] =
          (float)(sum / (double)avgLength);
    }
  }
}

/*
This is a general moving average filter for a single long array.

The design goal is to use shared memory in a way that allows each thread
in the warp to ALWAYS use a unique shared memory bank.

E.g.
N = 31 (length of moving average window)

Thread 0 | Thread 1 | Thread 2 ...
0->30    | 33->63   | 66->96
1->31    | ...
...
32->62   |

As can be seen above, thread (i) starts on the i'th bank and ends on the i'th
bank. On the first iteration, as the threads start reading forwards, each thread
is always accessing a different bank, maximising our shared memory access.

Doing it this way allows each thread to operate independently, and allows it to
optimise the moving average in the standard (add new element, subtract oldest
element) way, instead of having to synchronize with each other.

Note that in the above scenario, each thread outputs 33 elements.
This is not a required number! Other values are possible:

E.g.
NUM_PER_THREAD = 31

Thread 0    | Thread 1     | Thread 2 ...
0->30  (B00)| 31->61 (B31) | 62->92  (B28)
1->31  (B01)| 32->62 (B00) |
...
30->60 (B30)| 61->91 (B29) | 92->102 (B26)

However, some values should be avoided, like 32 (for obvious reasons),
but also all even numbers.
Odd numbers will always exactly fill all available shared memory banks, whereas
even numbers will always overlap in some memory banks.
Simple check is via:

```
uset = set(np.arange(32))
for i in range(1,34):
    s = (np.arange(32)*i) % 32 # Defines the bank index for each thread
    if set(s) != uset:
        print(i)
```
*/
extern "C" __global__ void movingAverage(const float *d_x, const int xlen,
                                         const int avgLength,
                                         const int NUM_PER_THREAD,
                                         float *d_out, // should also be xlen
                                         const bool sumInstead) {
  // Total length we have to read in from global mem
  int workspaceInputLength = blockDim.x * NUM_PER_THREAD + avgLength - 1;
  // And to output to global mem
  int workspaceOutputLength = blockDim.x * NUM_PER_THREAD;
  // Where we start reading from global mem for this block
  int i0 =
      workspaceOutputLength * blockIdx.x - (avgLength - 1); // offset backwards
  // Where to start writing to global mem for this block
  int o0 = workspaceOutputLength * blockIdx.x;

  // allocate shared memory
  extern __shared__ double s[];

  float *s_x = (float *)s; // (workspaceInputLength) floats
  float *s_ws =
      (float *)&s_x[workspaceInputLength]; // (workspaceOutputLength) floats

  for (int t = threadIdx.x; t < workspaceInputLength; t += blockDim.x) {
    // Only read if it's in range
    int idx = t + i0;
    if (idx >= 0 && idx < xlen)
      s_x[t] = d_x[idx];
    else
      s_x[t] = 0;
  }
  __syncthreads();

  // Now compute the 'first' average for each thread
  double sum =
      0.0; // We internally calculate as a double to have higher precision
  for (int i = 0; i < avgLength; i++) {
    sum += (double)s_x[threadIdx.x * NUM_PER_THREAD + i];
  }
  s_ws[threadIdx.x * NUM_PER_THREAD] =
      sumInstead ? (float)sum : (float)(sum / (double)avgLength);

  // For the rest of it, compute the average by removing earliest and adding
  // latest
  for (int j = 1; j < NUM_PER_THREAD; j++) {
    sum -= (double)s_x[threadIdx.x * NUM_PER_THREAD + j - 1];
    sum += (double)s_x[threadIdx.x * NUM_PER_THREAD + avgLength + j - 1];
    // We either save the sum or the average
    s_ws[threadIdx.x * NUM_PER_THREAD + j] =
        sumInstead ? (float)sum : (float)(sum / (double)avgLength);
  }

  // Finally write our output coalesced to global
  for (int t = threadIdx.x; t < workspaceOutputLength; t += blockDim.x) {
    if (o0 + t < xlen)
      d_out[o0 + t] = s_ws[t];
  }
}

/*
This is very similar to the above kernel,
but developed specifically for complex values i.e. the accumulator is complex.

We make some other adjustments here, as the use-case is targeted at the
matrix profiler diagonal calculation:

1) We make this a __device__ function, for use in a bigger kernel.
Assumptions on shared memory layouts are to be determined.

2) We only calculate 'valid' inputs. Hence the first (few) blocks will not
internally zero-pad the input. This will also mean the maximum output length
is no longer (xlen), but rather (xlen - L + 1).

3) The default is now a sum, instead of an average. We remove the option for
averages.

4) The sum is calculated with a forward-looking window, rather than the causal
backward-looking window. Coupled with using only 'valid' inputs, this means that
the first output is the result of a sum from indices 0 to L-1 (inclusive). In
the previous kernel, this would have been indices -L+1 to 0 (inclusive).
*/
extern "C" __global__ void movingComplexSum(const complex<float> *d_x,
                                            const int xlen, const int avgLength,
                                            const int NUM_PER_THREAD,
                                            float *d_out, // should also be xlen
                                            const bool sumInstead) {
  // Lengths do not change..
  // Total length we have to read in from global mem
  int workspaceInputLength = blockDim.x * NUM_PER_THREAD + avgLength - 1;
  // And to output to global mem
  int workspaceOutputLength = blockDim.x * NUM_PER_THREAD;

  // Where we start reading from global mem for this block
  // This is now 0-offsetted
  int i0 = workspaceOutputLength * blockIdx.x;
  // Where to start writing to global mem for this block
  int o0 = workspaceOutputLength * blockIdx.x;

  // allocate shared memory
  extern __shared__ double s[];

  complex<float> *s_x = (complex<float> *)s; // (workspaceInputLength) floats
  complex<float> *s_ws =
      (complex<float>
           *)&s_x[workspaceInputLength]; // (workspaceOutputLength) floats

  for (int t = threadIdx.x; t < workspaceInputLength; t += blockDim.x) {
    // Only read if it's in range
    int idx = t + i0;
    if (idx >= 0 && idx < xlen)
      s_x[t] = d_x[idx];
    else
      s_x[t] = 0;
  }
  __syncthreads();

  // Now compute the 'first' average for each thread
  complex<double> sum =
      0; // We internally calculate as a double to have higher precision
  for (int i = 0; i < avgLength; i++) {
    sum += (complex<double>)s_x[threadIdx.x * NUM_PER_THREAD + i];
  }
  s_ws[threadIdx.x * NUM_PER_THREAD] =
      sumInstead ? (complex<float>)sum
                 : (complex<float>)(sum / (double)avgLength);

  // For the rest of it, compute the average by removing earliest and adding
  // latest
  for (int j = 1; j < NUM_PER_THREAD; j++) {
    sum -= (double)s_x[threadIdx.x * NUM_PER_THREAD + j - 1];
    sum += (double)s_x[threadIdx.x * NUM_PER_THREAD + avgLength + j - 1];
    // We either save the sum or the average
    s_ws[threadIdx.x * NUM_PER_THREAD + j] =
        sumInstead ? (float)sum : (float)(sum / (double)avgLength);
  }

  // Finally write our output coalesced to global
  for (int t = threadIdx.x; t < workspaceOutputLength; t += blockDim.x) {
    if (o0 + t < xlen)
      d_out[o0 + t] = s_ws[t];
  }
}
