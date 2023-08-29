import numpy as np
from xcorrRoutines import *

data = np.arange(200).astype(np.float32).view(np.complex64)
print(data)

print(np.linalg.norm(data[9:19]))
print(np.linalg.norm(data[10:20]))
print(np.linalg.norm(data[11:21]))

print(np.linalg.norm(data[69:81]))
print(np.linalg.norm(data[70:82]))
print(np.linalg.norm(data[71:83]))

print(np.linalg.norm(data[9:19])**2 + np.linalg.norm(data[69:81])**2)
print(np.linalg.norm(data[10:20])**2 + np.linalg.norm(data[70:82])**2)
print(np.linalg.norm(data[11:21])**2 + np.linalg.norm(data[71:83])**2)

print(np.abs(np.vdot(data[10:20], data[10:20]) + np.vdot(data[70:82], data[70:82]))**2)

gxc = GroupXcorrCZT(
    data, np.array([10, 70]), np.array([10, 12]), 
    -50.0, 50.0, 50.0, 1000
)

print(gxc.ystack)
print(gxc.ystackNormSq)

results = gxc.xcorr(data, np.arange(9,12))
print(results)
