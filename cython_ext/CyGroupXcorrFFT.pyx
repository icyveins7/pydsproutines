# distutils: language = c++
import numpy as np
cimport numpy as np
from GroupXcorrFFT cimport GroupXcorrFFT, Ipp32fc, Ipp64f, bool

cdef class CyGroupXcorrFFT:
    cdef GroupXcorrFFT* gxcfft
    
    def __cinit__(self,
                  np.ndarray[np.complex64_t, ndim=2] ygroups,
                  np.ndarray[np.int32_t, ndim=1] offsets,
                  int fs, int fftlen=-1,
                  bool autoConj=True):
        
        self.gxcfft = new GroupXcorrFFT(
            <Ipp32fc*>ygroups.data, <int>ygroups.shape[0], <int>ygroups.shape[1],
            <int*>offsets.data,
            fs, fftlen,
            autoConj
        )
        
    def __dealloc__(self):
        del self.gxcfft