from filterRoutines import *
from timingRoutines import Timer

timer = Timer()

# Create the data
rows = 5000
cols = 2000
x = cp.random.randn(rows*cols).astype(np.float32)

# And taps
avgLength = 100
avgTaps = cp.ones(avgLength, dtype=np.float32)/avgLength


# Run something very long as a whole using our filter function
cpl = cp_lfilter(avgTaps, x)
timer.start()
cpl = cp_lfilter(avgTaps, x) # Run second time for warmup
h_cpl = cpl.get()
timer.end("cp_lfilter/cpsps.convolve")


# # Run using our custom kernels? TODO: custom kernels only work with complex64 now
# cpkf = CupyKernelFilter()
# cpk = cpkf.filter_smtaps(
#     x, avgTaps
# )
# timer.start()
# cpk = cpkf.filter_smtaps(
#     x, avgTaps
# )
# h_cpk = cpk.get()
# timer.end()


# Run it row-wise
rowMovAvg = cupyMultiMovingAverage(x.reshape((rows,cols)), avgLength, THREADS_PER_BLOCK=128)
timer.start()
rowMovAvg = cupyMultiMovingAverage(x.reshape((rows,cols)), avgLength, THREADS_PER_BLOCK=128) # Run second time for warmup
h_rowMovAvg = rowMovAvg.get() # We time the copies as well to force the device synchronize
timer.end("movingAverageKernel")


