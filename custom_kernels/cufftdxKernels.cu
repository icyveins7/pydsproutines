/*
General notes:
These kernels are all highly templated and dependent on compiler options.
*/

#include <cufftdx.hpp>

using namespace cufftdx;

/*
Create a simple in-kernel FFT to do a user-defined FFT length and elements per thread.
These are defined in the compilation step and supplied as options to NVRTC via cupy's wrapper when loading the module.
Largely taken from the nvrtc_block example, with modifications.

The original example has a 64-element FFT that returns a (8,1,1) block dimensions for ElementsPerThread=8, 
as expected since 64/8=8.

Trying ElementsPerThread=16 validates this, which returns 64/16=4 i.e. (4,1,1). 
The shared memory does not change and is 512 bytes regardless i.e. only dependent on FFT size.

Running it with multiple blocks also works:
Example:
  4 EPT, 2 FFTS_PER_BLK,
  2 Total FFTs, length 64 each
  Invoke 1 block with the returned dimensions (16,2,1) -> note the .y == 2 because 2 per block.
  Results are as expected, just like a 'batched' FFT.
*/

// FFT Operators
using size_desc = Size<FFT_SIZE>; // supplied as -DFFT_SIZE
using fwddir_desc = Direction<fft_direction::forward>;
using bwddir_desc = Direction<fft_direction::inverse>; 
using c2ctype_desc = Type<fft_type::c2c>; // note that this does not dictate the precision!
using FFT = decltype(
    Block() + size_desc() + fwddir_desc() + c2ctype_desc() + Precision<float>() + 
    SM<700>() + // I fixed this for now but the original example has this as configurable as well
    ElementsPerThread<FFT_EPT>() + // supplied as -DFFT_EPT
    FFTsPerBlock<FFT_PER_BLK>() // supplied as -DFFT_PER_BLK, this changes the blockDim.y returned
);
using complex_type = typename FFT::value_type;

__constant__ dim3         fft_block_dim     = FFT::block_dim; // get these with RawModule.get_global()
__constant__ unsigned int fft_shared_memory = FFT::shared_memory_size;
__constant__ bool         fft_requires_workspace = FFT::requires_workspace;
__constant__ unsigned int fft_stride = FFT::stride;

inline __device__ unsigned int batch_offset(const unsigned int local_fft_id,
                                            const unsigned int ffts_per_block = blockDim.y) { // this default value is in fact how the blockDim changes when you change the FFTS_PER_BLK
    unsigned int global_fft_id = ffts_per_block == 1 ? blockIdx.x : (blockIdx.x * ffts_per_block + local_fft_id);
    return cufftdx::size_of<FFT>::value * global_fft_id;
}

// Test kernel (with my annotations)
extern "C" __global__ void test_kernel(
    typename FFT::value_type* input // this works with cp.complex64 as expected!
)
{
  typename FFT::value_type thread_data[FFT::storage_size]; // this is a thread-local register

  const unsigned int offset = batch_offset(threadIdx.y, FFT::ffts_per_block);
  constexpr unsigned int stride = FFT::stride; // this seems to be blockDim.x? not sure if it ever changes
  unsigned int index = offset + threadIdx.x; // this is where in the global mem input to start reading from

  // This is filling in the input data on the register
  // it seems like the format goes from 0,1,2,....
  // to 
  // Thread   0:   0,    m,   2m, ..., (n-1)*m
  // Thread   1:   1,  m+1, 2m+1, ...
  // ................
  // Thread m-1, m-1, 2m-1, ....
  // where m is the number of threads in the block,
  // and n is the number of elements per thread
  // Seems like m*n = fft length?
  for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
    if ((i * stride + threadIdx.x) < cufftdx::size_of<FFT>::value) {
        thread_data[i] = input[index]; // the global memory reads are actually coalesced since 'index' is incrementing for each thread
        index += stride;
    }
  }

  extern __shared__ FFT::value_type shared_mem[]; // this is the shared memory requirement that is compile-time defined above
  FFT().execute(thread_data, shared_mem);

  index = offset + threadIdx.x;
  for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
    if ((i * stride + threadIdx.x) < cufftdx::size_of<FFT>::value) {
        input[index] = thread_data[i];
        index += stride;
    }
  }
}


/*
================================================================
================================================================
================================================================
                  FILTER FLAVOURS
================================================================
================================================================
================================================================
*/

/*
The goal for this kernel is to replicate the output of our previous filter implementation,
which stored both the raw taps and an appropriate section of the input data, and performed the naive multiplications in shared memory.

Here we use a convolution within the kernel with an in-kernel FFT and IFFT, again
operating on 1 section only per block. That is, only 1 FFT and 1 IFFT are called per block.
This lets us reuse a lot of the example code from the cufftdx convolution.cu kernel.
*/

using IFFT = decltype(
    Block() + size_desc() + bwddir_desc() + c2ctype_desc() + Precision<float>() + 
    SM<700>() + // I fixed this for now but the original example has this as configurable as well
    ElementsPerThread<FFT_EPT>() + // supplied as -DFFT_EPT
    FFTsPerBlock<FFT_PER_BLK>() // supplied as -DFFT_PER_BLK, this changes the blockDim.y returned
);

extern "C" __global__ void filter_sm_convolution(
  const complex_type* input, int input_len,
  const complex_type* taps_fft, int tapslen,
  complex_type* output
){
  // Local array requirement for cufftdx
  complex_type thread_data[FFT::storage_size]; // this is a thread-local register

  // Define the ACTUAL convolved length based on FFT size: L + N - 1 = FFT size
  const int length_per_blk = FFT_SIZE - tapslen + 1;
  if (length_per_blk < 0) // Should never happen!
    return;
  // Define the first index to read (including zero-padded at the front of input, so we can go negative)
  int writeIdx = blockIdx.x * length_per_blk;
  int readIdx = writeIdx - tapslen + 1 + threadIdx.x; // offset backwards so we have the 'delay'

  // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
  const unsigned int local_fft_id = threadIdx.y;
  // Load data from global memory to registers
  // example::io<FFT>::load(data, thread_data, local_fft_id);

  // TODO: fix reading? not even sure
  for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
    if ((i * FFT::stride + threadIdx.x) < cufftdx::size_of<FFT>::value) 
    {
      // Read from the global memory if in range
      if (readIdx >= 0 && readIdx < input_len)
        thread_data[i].x = readIdx; // input[readIdx];
      // else
      // {
      //   thread_data[i].x = taps_fft[i].x;
      //   thread_data[i].y = FFT_SIZE;
      // }

      readIdx += FFT::stride;
    }
  }

  // // Execute FFT
  // extern __shared__ complex_type shared_mem[];
  // FFT().execute(thread_data, shared_mem);

  // // Multiply by the taps_fft
  // // cast doesn't seem to work?
  // // complex<float> *cast_thread_data = reinterpret_cast<complex<float>*>(thread_data); // cast it so we can use cupy's internal multiply functions
  // for (unsigned int i = 0; i < FFT::elements_per_thread; i++) 
  // {
  //   cast_thread_data[i].x = cast_thread_data[i].x * taps_fft[i * FFT::stride + threadIdx.x].x - cast_thread_data[i].y * taps_fft[i * FFT::stride + threadIdx.x].y;
  //   cast_thread_data[i].y = cast_thread_data[i].x * taps_fft[i * FFT::stride + threadIdx.x].y + cast_thread_data[i].y * taps_fft[i * FFT::stride + threadIdx.x].x;
  // }

  // // Execute inverse FFT
  // IFFT().execute(thread_data, shared_mem);

  // Save results
  // example::io<FFT>::store(thread_data, data, local_fft_id);

  for (unsigned int i = 0; i < FFT::elements_per_thread; i++) 
  {
    // Set the read index from the output (skip the front section which should have convolved with 0s/circular)
    readIdx = i * FFT::stride + threadIdx.x;

    if (readIdx < cufftdx::size_of<FFT>::value) //  && readIdx >= tapslen - 1) 
    {
        output[writeIdx] = thread_data[i];
        writeIdx += FFT::stride;
    }
  }
}