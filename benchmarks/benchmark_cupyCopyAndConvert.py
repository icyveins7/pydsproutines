# Simple script to test copying int16s then converting to float32s on-device
# as opposed to converting to float32s then copying

import numpy as np
import cupy as cp
from timingRoutines import Timer

timer = Timer()

# Create a test array
length = 100000000
data = (np.arange(length) % 256).astype(np.int16)


# Test 1, convert on CPU then copy to GPU
timer.start()
fdata = data.astype(np.float32) # ~70ms from python timer
timer.evt("convert to float32")
d_fdata = cp.asarray(fdata) # 51.4ms from nsys timing (python timer not accurate for this)
timer.end()

# Test 2, move to GPU then convert on device
# This should be faster on copy, but has overhead of requiring additional GPU memory
d_data = cp.asarray(data) # 26.6ms, about half as expected
d_fdata = d_data.astype(np.float32) # 1.7ms, very small

# Repeat it again
d_data = cp.asarray(data)
d_fdata = d_data.astype(np.float32) 
# Note that on the first invocation of the .astype(np.float32), 
# nsys will show that it starts way after the memcpy
# This is probably due to the jit compile of the kernel
# This second call shows up in nsys as right after the memcpy