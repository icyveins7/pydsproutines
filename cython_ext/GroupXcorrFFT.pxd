from libcpp.vector cimport vector
from libcpp cimport bool

# Declare the class with cdef
cdef extern from "GroupXcorrFFT.cpp":
    pass

cdef extern from "GroupXcorrFFT.h":
    ctypedef struct Ipp32fc:
        pass
    
    ctypedef double Ipp64f
    
    ctypedef float Ipp32f
    
    ctypedef unsigned char Ipp8u
    
    cdef cppclass GroupXcorrFFT:
        GroupXcorrFFT(Ipp32fc*, int, int, int*, int, int, bool) except +
        
        void xcorr(Ipp32fc*, int, int*, int, Ipp32f*, int)