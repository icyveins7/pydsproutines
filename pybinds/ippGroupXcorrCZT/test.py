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

print(np.exp(-1j*2*np.pi*np.array([-0.1,0,0.1])/100*60))

gxc = GroupXcorrCZT(
    data, np.array([10, 70]), np.array([10, 12]), 
    -0.1, 0.1, 0.1, 100
)

print(gxc.ystack)
print(gxc.ystackNormSq)

results, cztfreq = gxc.xcorr(data, np.arange(9,12))
print(results)

import sys
if sys.platform.startswith('win32'):
    import os
    os.add_dll_directory(os.path.join(os.environ['IPPROOT'], "redist", "intel64"))

from pbIppGroupXcorrCZT import pbIppGroupXcorrCZT

# Create the pybind obj
# pbgxc = pbIppGroupXcorrCZT(12, -0.1, 0.1, 0.1, 100)
# print(pbgxc)
# print("pybind numthreads = %d" % (pbgxc.getNumThreads()))

# NOTE: Creating more than 1 object causes crashes TODO: fix this
pbgxc4 = pbIppGroupXcorrCZT(12, -0.1, 0.1, 0.1, 100, 1)
print(pbgxc4)
print("pybind Numthreads = %d" % (pbgxc4.getNumThreads()))

pbgxc4 = pbIppGroupXcorrCZT(12, -0.1, 0.1, 0.1, 100, 4)
print(pbgxc4)
print("pybind Numthreads = %d" % (pbgxc4.getNumThreads()))



# Add the groups
pbgxc4.addGroup(
    0, data[10:20]
)
pbgxc4.addGroup(
    60, data[70:82]
)
try:
    pbgxc4.addGroup(
        65, data[75:85]
    )
except Exception as e:
    print("Expected error: %s" % str(e))


pbgxc4.printGroups()

pbresults = pbgxc4.xcorr(data, 9, 1, 3)
print(pbresults)
# print(type(pbresults))


#%% Clear the groups and use the convenience method to add instead
pbgxc4.resetGroups()
pbgxc4.addGroupsFromArray(
    np.array([10, 70]), np.array([10, 12]), data
)
pbgxc4.printGroups()
pbresults = pbgxc4.xcorr(data, 9, 1, 3)
print(pbresults)


#%%
from verifyRoutines import *
# print(pbresults.shape)
# print(results.shape)
compareValues(results.flatten(), pbresults.flatten())



