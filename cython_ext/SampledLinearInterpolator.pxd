# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 18:59:09 2021

@author: Seo
"""

# Declare the class with cdef
cdef extern from "SampledLinearInterpolator.cpp":
    pass

cdef extern from "SampledLinearInterpolator.h":
    
    cdef cppclass SampledLinearInterpolator_64f:
        SampledLinearInterpolator_64f(double*, double*, int, double) except +
        
        void lerp(double*, double*, int)