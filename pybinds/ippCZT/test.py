import sys
if sys.platform.startswith('win32'):
    import os
    os.add_dll_directory(os.path.join(os.environ['IPPROOT'], "redist", "intel64"))

from pbIppCZT32fc import pbIppCZT32fc
import numpy as np

from spectralRoutines import *
from verifyRoutines import *

x = np.random.rand(10) + np.random.rand(10)*1j
x = x.astype(np.complex64) # Cast


length = x.size
f1 = -1000.0
f2 = 1000.0
fstep = 1.0
fs = float(length)

for i in range(x.size):
    x[i] = i + i*1j


pbczt = pbIppCZT32fc(x.size, f1, f2, fstep, fs)
y = pbczt.run(x)
xc = np.array(x[:9], copy=True)
try:
    pbczt.run(xc)
except Exception as e:
    print("Expected error for wrong length")
# print(y)

cztobj = CZTCached(x.size, f1, f2, fstep, fs, convertTo32fc=True)
print(cztobj.fv)
print(cztobj.aa)
print(cztobj.ww)
yn = cztobj.run(x)

compareValues(y, yn)

#%% Try against a 2D
xl = np.zeros((2, x.size), dtype=np.complex64)
xl[0] = x[:]
xl[1] = x[:]

# Show that running with run() on each row works
for i in range(xl.shape[0]):
    y = pbczt.run(xl[i,:])
    compareValues(y, yn)

# Now show that we can use runMany()
yl = pbczt.runMany(xl)
compareValues(yl[0], yn)
compareValues(yl[1], yn)

# Show error when wrong dimensions

try:
    pbczt.runMany(xl[0,:])
except Exception as e:
    print("Expected error for wrong dimensions")

