# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 18:59:09 2021

@author: Seo
"""

# Declare the class with cdef
cdef extern from "SampledLinearInterpolator.cpp":
    pass

cdef extern from "SampledLinearInterpolator.h":
    ctypedef struct Ipp64fc:
        pass
		
    cdef cppclass SampledLinearInterpolatorWorkspace_64f:
        SampledLinearInterpolatorWorkspace_64f(int) except +
    
    cdef cppclass SampledLinearInterpolator_64f:
        SampledLinearInterpolator_64f(double*, int, double) except +
        
        void lerp(double*, double*, int, SampledLinearInterpolatorWorkspace_64f*)
        
    cdef cppclass ConstAmpSigLerp_64f:
        ConstAmpSigLerp_64f(double, double, double*, int, double, double, double) except +
        
        void propagate(double*, double*, double, int, Ipp64fc*, SampledLinearInterpolatorWorkspace_64f*)
        
    cdef cppclass ConstAmpSigLerpBursty_64f:
        ConstAmpSigLerpBursty_64f() except +
        
        void addSignal(ConstAmpSigLerp_64f*)
        void propagate(double*, double*, double*, double*, int, Ipp64fc*, SampledLinearInterpolatorWorkspace_64f*)
        
    cdef cppclass ConstAmpSigLerpBurstyMulti_64f:
        ConstAmpSigLerpBurstyMulti_64f() except +
        
        void addSignal(ConstAmpSigLerpBursty_64f*)
        void propagate(double*, double*, double*, double*, int, int, Ipp64fc*, int)
        
        