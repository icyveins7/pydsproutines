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

    import os
    os.add_dll_directory(os.path.join(os.environ['IPPROOT'], "redist", "intel64"))
    from pbIppCZT32fc import pbIppCZT32fc
    
    timer = Timer()
    
    length = 10000
    f1 = -1000.0
    f2 = 1000.0
    fstep = 1.0
    fs = length
    
    x = np.vstack([randnoise(length, 1, 1, 10).astype(np.complex64) for i in range(10)])
    
    #%% GPU based CZT object, using cupy functions
    try:
        import cupy as cp
        d_cztobj = CZTCachedGPU(length, f1, f2, fstep, fs)
        d_x = cp.asarray(x)
        
        timer.start()
        e1 = cp.cuda.get_current_stream().record()
        d_out = d_cztobj.runMany(d_x)
        e2 = cp.cuda.get_current_stream().record()
        e2.synchronize()
        print("cuda events: %f ms" % (cp.cuda.get_elapsed_time(e1, e2)))
        timer.evt("czt gpu batch")

        #%% Test single CZT against custom kernel
        d_src = d_x[0,:]

        e1 = cp.cuda.get_current_stream().record()
        
        d_out = d_cztobj.run(d_src)

        e2 = cp.cuda.get_current_stream().record()
        e2.synchronize()
        print("cuda events (single czt): %f ms" % (cp.cuda.get_elapsed_time(e1, e2)))

        e1 = cp.cuda.get_current_stream().record()

        d_outkernel_interrim = cupyDotTonesScaling(-f1/fs, -fstep/fs, d_cztobj.getFreq().size, d_src) # Remember, you need to put minus sign to compare with CZTs
        d_outkernel = cp.sum(d_outkernel_interrim, axis=0)

        e2 = cp.cuda.get_current_stream().record()
        e2.synchronize()
        print("cuda events (single czt custom kernel): %f ms" % (cp.cuda.get_elapsed_time(e1, e2)))

        
    except ImportError:
        print("cupy not installed, skipping czt gpu test")
    
    #%% CPU based CZT object
    cztobj = CZTCached(length, f1, f2, fstep, fs, True)
    out = cztobj.runMany(x)
    timer.evt("czt cpu batch")

    #%% CPU based pre-compiled IPP CZT object
    pbczt = pbIppCZT32fc(length, f1, f2, fstep, float(length))
    # pbczt = pbIppCZT32fc(length, -0.1, 0.1, 0.01, 1.0)
    print("pbCzt ok")
    for i in range(x.shape[0]):
        xc = np.array(x[i,:], dtype=np.complex64)
        print(xc)
        print(xc.dtype)
        print(xc.size)
        print(pbczt.m_N)
        pbout = pbczt.run(xc)
        print("pbCZT %d" % i)

    #%%
    timer.end()

    #%%
    # rawChg, fracChg = compareValues(d_out.get().flatten(), out.flatten())
    # assert(fracChg < 1e-2)

    # h_out = cztobj.run(d_src.get())

    # rawChg, fracChg = compareValues(h_out, d_outkernel.get())

    

    