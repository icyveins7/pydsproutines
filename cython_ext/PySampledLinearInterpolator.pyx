# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 19:04:03 2021

@author: Seo
"""

# distutils: language = c++

import numpy as np
cimport numpy as np

from SampledLinearInterpolator cimport SampledLinearInterpolatorWorkspace_64f, SampledLinearInterpolator_64f
from SampledLinearInterpolator cimport ConstAmpSigLerp_64f, ConstAmpSigLerpBursty_64f, ConstAmpSigLerpBurstyMulti_64f
from SampledLinearInterpolator cimport Ipp64fc

cdef class PySampledLinearInterpolatorWorkspace_64f:
    cdef SampledLinearInterpolatorWorkspace_64f *ws
    
    def __cinit__(self, int size):
        self.ws = new SampledLinearInterpolatorWorkspace_64f(size)
        
    def __dealloc__(self):
        del self.ws

cdef class PySampledLinearInterpolator_64f:
    cdef SampledLinearInterpolator_64f* sli
    
    def __cinit__(self, np.ndarray[np.float64_t, ndim=1] y, double T):
        
        # print("Attempting to init..")
        self.sli = new SampledLinearInterpolator_64f(<double*>y.data, <int>y.size, T)
        
    def __dealloc__(self):
        del self.sli
    
    def lerp(self, np.ndarray[np.float64_t, ndim=1] xq, PySampledLinearInterpolatorWorkspace_64f pws):
        cdef np.ndarray yq = np.zeros(xq.size, np.float64)
        self.sli.lerp(<double*>xq.data, <double*>yq.data, xq.size, 
                      <SampledLinearInterpolatorWorkspace_64f*>pws.ws)
    
        return yq
        
cdef class PyConstAmpSigLerp_64f:
    cdef ConstAmpSigLerp_64f* sig
    
    def __cinit__(self, np.ndarray[np.float64_t, ndim=1] timevec,
                        np.ndarray[np.float64_t, ndim=1] phasevec,
                        double T, double amp, double fc):
    
        assert(timevec.size == phasevec.size)
        self.sig = new ConstAmpSigLerp_64f(<double>timevec[0], <double>timevec[-1], <double*>phasevec.data, 
                                            <int>timevec.size, T, amp, fc)
                                            
    def __dealloc__(self):
        del self.sig
        
    def propagate(self, np.ndarray[np.float64_t, ndim=1] t,
                        np.ndarray[np.float64_t, ndim=1] tau,
                        double phi,
                        PySampledLinearInterpolatorWorkspace_64f pws,
                        int startIdx=-1):
                        
        assert(t.size == tau.size)
        cdef np.ndarray x = np.zeros(t.size, np.complex128)
        self.sig.propagate(<double*>t.data, <double*>tau.data, phi, t.size, <Ipp64fc*>x.data,
                           <SampledLinearInterpolatorWorkspace_64f*>pws.ws, startIdx)
        
        return x
    
cdef class PyConstAmpSigLerpBursty_64f:
    cdef ConstAmpSigLerpBursty_64f* sigb
    cdef int _numBursts
    
    def __cinit__(self):
        self.sigb = new ConstAmpSigLerpBursty_64f()
        self._numBursts = 0
        
    def __dealloc__(self):
        del self.sigb
        
    def numBursts(self):
        return self._numBursts
        
    def addSignal(self, PyConstAmpSigLerp_64f pycasl):
        self.sigb.addSignal(<ConstAmpSigLerp_64f*>pycasl.sig)
        self._numBursts = self._numBursts + 1
        
    def propagate(self, np.ndarray[np.float64_t, ndim=1] t,
                        np.ndarray[np.float64_t, ndim=1] tau,
                        np.ndarray[np.float64_t, ndim=1] phiArr,
                        np.ndarray[np.float64_t, ndim=1] tJumpArr,
                        PySampledLinearInterpolatorWorkspace_64f pws,
                        int startIdx=-1):
        
        assert(t.size==tau.size)
        assert(phiArr.size == tJumpArr.size)
        assert(phiArr.size == self._numBursts)
        
        cdef np.ndarray x = np.zeros(t.size, np.complex128)
        self.sigb.propagate(<double*>t.data, <double*>tau.data, <double*>phiArr.data, <double*>tJumpArr.data,
                            t.size, <Ipp64fc*>x.data,
                            <SampledLinearInterpolatorWorkspace_64f*>pws.ws,
                            startIdx)
        
        return x


cdef class PyConstAmpSigLerpBurstyMulti_64f:
    cdef ConstAmpSigLerpBurstyMulti_64f* sigbm
    cdef int numSig
    cdef int numBursts
    
    def __cinit__(self):
        self.sigbm = new ConstAmpSigLerpBurstyMulti_64f()
        self.numBursts = -1
        self.numSig = 0

    def __dealloc__(self):
        del self.sigbm
        
    def addSignal(self, PyConstAmpSigLerpBursty_64f pycaslb):
        self.sigbm.addSignal(pycaslb.sigb)
        if (self.numBursts == -1):
            self.numBursts = pycaslb.numBursts()
        else:
            assert(self.numBursts == pycaslb.numBursts()) # make sure they are all the same number of bursts
            
        self.numSig = self.numSig + 1
        # print("Total signals: %d" % (self.numSig))
        
    def propagate(self, np.ndarray[np.float64_t, ndim=1] t,
                        np.ndarray[np.float64_t, ndim=1] tau,
                        np.ndarray[np.float64_t, ndim=1] phiArrs,
                        np.ndarray[np.float64_t, ndim=1] tJumpArrs,
                        int numThreads):
        
        assert(t.size == tau.size)
        assert(phiArrs.size == tJumpArrs.size)
        assert(phiArrs.size == self.numBursts * self.numSig)
        
        cdef np.ndarray x = np.zeros(t.size, np.complex128)

        self.sigbm.propagate(<double*>t.data, <double*>tau.data, <double*>phiArrs.data, <double*>tJumpArrs.data,
                            self.numBursts, t.size, <Ipp64fc*>x.data, numThreads)
        
        return x
