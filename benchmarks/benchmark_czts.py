# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 17:44:12 2023

@author: lken
"""

if __name__ == "__main__":
    from spectralRoutines import *
    from signalCreationRoutines import *
    
    length = 1000000
    f1 = -1000.0
    f2 = 1000.0
    fstep = 0.001
    fs = 10000000
    
    x = randnoise(length, 1, 1, 10).astype(np.complex64)
    
    cztobj = CZTCachedGPU(length, f1, f2, fstep, fs)
    d_x = cp.asarray(x)
    
    out = cztobj.run(d_x)

