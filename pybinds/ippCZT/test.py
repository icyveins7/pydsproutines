import os
os.add_dll_directory(os.path.join(os.environ['IPPROOT'], "redist", "intel64"))

from pbIppCZT32fc import pbIppCZT32fc
import numpy as np

from spectralRoutines import *
from verifyRoutines import *

x = np.random.rand(10) + np.random.rand(10)*1j
x = x.astype(np.complex64) # Cast

pbczt = pbIppCZT32fc(x.size, -0.1, 0.1, 0.01, 1)
print("Instantiated object.")
# print(pbczt.m_ww)

for i in range(3): # Repeat a few times to make sure no crashes
    y = pbczt.run(x)
    assert(np.all(~np.isnan(y)))
print(y)

print("Check against python method.")
cztobj = CZTCached(x.size, -0.1, 0.1, 0.01, 1, convertTo32fc=True)
yn = cztobj.run(x)
print(yn)


length = x.size
f1 = -1000.0
f2 = 1000.0
fstep = 1.0
fs = float(length)

try:
    pbczt = pbIppCZT32fc(x.size, f1, f2, fstep, fs)
    y = pbczt.run(x)

    cztobj = CZTCached(x.size, f1, f2, fstep, fs, convertTo32fc=True)
    yn = cztobj.run(x)

except Exception as e:
    print(e)



compareValues(y, yn)

