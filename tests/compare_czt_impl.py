from spectralRoutines import *
from verifyRoutines import *

#%%
x = np.random.randn(10).view(np.complex128)
# print('x', x)

f1 = -0.1
f2 = 0.1
binWidth = 0.01
fs = 1.0

#%% First check against scipy
ys = czt_scipy(x, f1, f2, binWidth, fs)

#%% Then ours
# y = czt(x, f1, f2, binWidth, fs)
c = CZTCached(x.size, f1, f2, binWidth, fs)
y = c.run(x)
print('nfft = %d' % (c.nfft))
print('m + k = %d' % (c.m + c.k))

#%% Final check by doing it against dft
y_dft = dft(x, np.arange(f1, f2+binWidth, binWidth), fs)

#%% Check
print("custom vs scipy")
rawChg, fracChg = compareValues(y, ys)
print(rawChg, fracChg)

#%%
print("custom vs dft")
rawChg, fracChg = compareValues(y, y_dft)
print(rawChg, fracChg)