from libcpp.vector cimport vector
from libcpp cimport bool

# Declare the class with cdef
cdef extern from "IppXcorrFFT.cpp":
    pass

cdef extern from "IppXcorrFFT.h":
    ctypedef struct Ipp32fc:
        pass

    cdef cppclass IppXcorrFFT_32fc:
        IppXcorrFFT_32fc(Ipp32fc*, int, int, bool) except +

        void xcorr_array(Ipp32fc*, int, int, int, int, float*, int*, int)
    
        vector[float] m_productpeaks
        vector[int] m_freqlistinds