#include <cupy/complex.cuh>


/*
Design goal:
Linearly interpolate an input vector based on its sample duration
and a target sample duration.

TODO
*/
extern "C"
__global__ void lerpConstantSampleRate(
    const complex<float> *input, 
    const int inputLength,
    const double T0, // this is the sample duration of input
    complex<float> *output, 
    const int outputRows, 
    const int outputLength,
    const double T, // this is the sample duration for output
    const int* offsets, // each outputRow corresponds to an offset
    const int outputPerBlock // number of outputs to calculate per block
){
    // Define the first output index
    const int oIdx0 = blockIdx.x * outputPerBlock;
    // Calculate the start time for this first index
}