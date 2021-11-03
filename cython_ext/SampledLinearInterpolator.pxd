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
    
    cdef cppclass SampledLinearInterpolator_64f:
        SampledLinearInterpolator_64f(double*, double*, int, double) except +
        
        void lerp(double*, double*, int)
        
    cdef cppclass ConstAmpSigLerp_64f:
        ConstAmpSigLerp_64f(double*, double*, int, double, double, double) except +
        
        void propagate(double*, double*, double, int, Ipp64fc*)
        