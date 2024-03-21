import cupy as cp

from timingRoutines import Timer

length = 10000000
data = cp.random.randn(length*2).astype(cp.float64).view(cp.complex128).astype(cp.complex64)

loops = 2
timer = Timer()
timer.start()
for i in range(loops):
    out = cp.fft.fft(data)

# Then compare time to running many small FFTs
smalllen = 200
rsdata = data.reshape((-1,200))
for i in range(loops):
    out = cp.fft.fft(rsdata, axis=1) 

total = timer.end()
print("%fs per loop" % (total/loops))
