# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 19:04:03 2021

@author: Seo
"""

# distutils: language = c++

import numpy as np
cimport numpy as np

from SampledLinearInterpolator cimport SampledLinearInterpolator_64f, ConstAmpSigLerp_64f, Ipp64fc

cdef class PySampledLinearInterpolator_64f:
    cdef SampledLinearInterpolator_64f* sli
    
    def __cinit__(self, np.ndarray[np.float64_t, ndim=1] x, np.ndarray[np.float64_t, ndim=1] y, double T):
        
        # print("Attempting to init..")
        assert(x.size == y.size)
        self.sli = new SampledLinearInterpolator_64f(<double*>x.data, <double*>y.data, <int>x.size, T)
        
    def __dealloc__(self):
        del self.sli
    
    def lerp(self, np.ndarray[np.float64_t, ndim=1] xq):
        cdef np.ndarray yq = np.zeros(xq.size, np.float64)
        self.sli.lerp(<double*>xq.data, <double*>yq.data, xq.size)
    
        return yq
        
cdef class PyConstAmpSigLerp_64f:
    cdef ConstAmpSigLerp_64f* sig
    
    def __cinit__(self, np.ndarray[np.float64_t, ndim=1] timevec,
                        np.ndarray[np.float64_t, ndim=1] phasevec,
                        double T, double amp, double fc):
    
        assert(timevec.size == phasevec.size)
        self.sig = new ConstAmpSigLerp_64f(<double*>timevec.data, <double*>phasevec.data, 
                                            <int>timevec.size, T, amp, fc)
                                            
    def __dealloc__(self):
        del self.sig
        
    def propagate(self, np.ndarray[np.float64_t, ndim=1] t,
                        np.ndarray[np.float64_t, ndim=1] tau,
                        double phi):
                        
        assert(t.size == tau.size)
        cdef np.ndarray x = np.zeros(t.size, np.complex128)
        self.sig.propagate(<double*>t.data, <double*>tau.data, phi, t.size, <Ipp64fc*>x.data)
        
        return x
