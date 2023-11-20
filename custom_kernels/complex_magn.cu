#include <cupy/complex.cuh>

/*
This is a simple grid stride kernel to use the cupy-inbuilt complex norm(),
which is actually magn squared, and not just magn() i.e. abs().
Evidence -> cupy/cupy/_core/include/cupy/complex/complex.h
*/
template <typename T, typename U>
__global__ 
void complex_magnSq_kernel(
    const complex<T> *x,
    int length,
    U *xMagnSq
) {
    int gridStride = gridDim.x * blockDim.x;
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < length; t += gridStride){
        xMagnSq[t] = (U)norm(x[t]); // cast to desired output type
    }
}

