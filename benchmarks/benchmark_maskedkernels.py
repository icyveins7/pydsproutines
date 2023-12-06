"""
This benchmark uses a few simplistic kernels to test whether the overhead of kernel launches is significant.
Assume we have a mask which denotes which row of a matrix to operate on.

Depending on the mask value, we can create a kernel which matches a certain value of the mask,
and then does a particular operation on those rows, leaving all other rows untouched.
Then we can repeat this for the other mask values. This is a simple approach, which constrains the kernel
to just having to think about the input for 1 mask value at a time. However, one might think that having to repeat the kernel
for all the different mask values in order to process all the rows may be too costly.

This benchmark proves otherwise. The extra kernel cost is likely to be insignificant, unless 
each kernel performs extremely little work. Here 'multiplyOnlyMaskedRows' is repeated twice,
with a separate input array for the 2 separate mask values.
'multiplyRowsBasedOnMask' takes in both input arrays for the 2 separate mask values, and is only invoked once.
There is only about 10 microseconds difference.

23.44%  970.75us         2  485.38us  485.12us  485.63us  multiplyOnlyMaskedRows
23.21%  961.35us         1  961.35us  961.35us  961.35us  multiplyRowsBasedOnMask
"""

from cupyExtensions import *

import cupy as cp

#%% Load the kernels
(multiplyOnlyMaskedRowsKernel,
multiplyRowsBasedOnMask), _ = cupyModuleToKernelsLoader("maskedaccess.cu", ["multiplyOnlyMaskedRows", "multiplyRowsBasedOnMask"])

#%% Initialize some data
M = 10000 # rows
N = 1000 # columns

x = cp.random.randn(2*M*N, dtype=cp.float32).view(cp.complex64).reshape((M,N))
y0 = cp.random.randn(2*M*N, dtype=cp.float32).view(cp.complex64).reshape((M,N))
y1 = cp.random.randn(2*M*N, dtype=cp.float32).view(cp.complex64).reshape((M,N))
out = cp.zeros_like(x)
mask = cp.random.randint(0,2,size=M,dtype=cp.int32)

# Execute kernel for value 0
multiplyOnlyMaskedRowsKernel(
    (M,),(128,),
    (mask,
    x,
    y0, # Use y0 for mask value 0
    out,
    N,
    0)
)

# Execute kernel for value 1
multiplyOnlyMaskedRowsKernel(
    (M,),(128,),
    (mask,
    x,
    y1, # Use y1 for mask value 1
    out,
    N,
    1)
)

# Execute merged kernel
multiplyRowsBasedOnMask(
    (M,),(128,),
    (mask,
    x,
    y0,
    y1,
    out,
    N)
)



