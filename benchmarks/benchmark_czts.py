# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 17:44:12 2023

@author: lken
"""

if __name__ == "__main__":
    from spectralRoutines import *
    from signalCreationRoutines import *
    from verifyRoutines import *
    from timingRoutines import Timer
    
    import cupy as cp
    
    timer = Timer()
    
    length = 1000000
    f1 = -1000.0
    f2 = 1000.0
    fstep = 0.001
    fs = 10000000
    
    x = np.vstack([randnoise(length, 1, 1, 10).astype(np.complex64) for i in range(10)])
    
    d_cztobj = CZTCachedGPU(length, f1, f2, fstep, fs)
    d_x = cp.asarray(x)
    
    timer.start()
    e1 = cp.cuda.get_current_stream().record()
    d_out = d_cztobj.runMany(d_x)
    e2 = cp.cuda.get_current_stream().record()
    print("cuda events: %f " % (cp.cuda.get_elapsed_time(e1, e2)))
    timer.evt("czt gpu batch")
    
    #%%
    cztobj = CZTCached(length, f1, f2, fstep, fs)
    out = cztobj.runMany(x)
    timer.evt("czt cpu batch")
    
    #%%
    timer.end()
