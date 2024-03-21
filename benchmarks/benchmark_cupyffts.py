import cupy as cp

from timingRoutines import Timer

length = 1000000
data = cp.random.randn(length*2).astype(cp.float64).view(cp.complex128).astype(cp.complex64)
# out = cp.zeros_like(data)

loops = 2
timer = Timer()
timer.start()
for i in range(loops):
    out = cp.fft.fft(data)

total = timer.end()
print("%fs per loop" % (total/loops))
