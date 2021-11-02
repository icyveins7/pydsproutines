# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 19:04:03 2021

@author: Seo
"""

# distutils: language = c++

import numpy as np
cimport numpy as np

from SampledLinearInterpolator cimport SampledLinearInterpolator_64f

cdef class PySampledLinearInterpolator_64f:
    cdef SampledLinearInterpolator_64f* sli
    
    def __cinit__(self, np.ndarray[np.float64_t, ndim=1] x, np.ndarray[np.float64_t, ndim=1] y, double T):
        
        print("Attempting to init..")
        assert(x.size == y.size)
        self.sli = new SampledLinearInterpolator_64f(<double*>x.data, <double*>y.data, <int>x.size, T)
        
    def __dealloc__(self):
        del self.sli
    
    def lerp(self, np.ndarray[np.float64_t, ndim=1] xq):
        cdef np.ndarray yq = np.zeros(xq.size, np.float64)
        self.sli.lerp(<double*>xq.data, <double*>yq.data, xq.size)
    
        return yq