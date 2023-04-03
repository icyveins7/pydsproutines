# distutils: language = c++
import numpy as np
cimport numpy as np
from GroupXcorrFFT cimport GroupXcorrFFT, Ipp32fc, Ipp32f, Ipp64f, bool

cdef class CyGroupXcorrFFT:
    """
    Cythonised version of GroupXcorrFFT.
    """

    cdef GroupXcorrFFT* gxcfft
    
    def __cinit__(self,
                  np.ndarray[np.complex64_t, ndim=2] ygroups,
                  np.ndarray[np.int32_t, ndim=1] offsets,
                  int fs, int fftlen=-1,
                  bool autoConj=True):
        """
        Instantiates a CyGroupXcorrFFT object.
        """
        
        self.gxcfft = new GroupXcorrFFT(
            <Ipp32fc*>ygroups.data, <int>ygroups.shape[0], <int>ygroups.shape[1],
            <int*>offsets.data,
            fs, fftlen,
            autoConj
        )
        
    def __dealloc__(self):
        del self.gxcfft
        
    # Main computation method
    def xcorr(self,
              np.ndarray[np.complex64_t, ndim=1] rx,
              np.ndarray[np.int32_t, ndim=1] shifts,
              int NUM_THREADS=1):
        """
        Main computation method.

        Parameters
        ----------
        rx : np.ndarray[np.complex64_t, ndim=1]
            Input array.
        shifts : np.ndarray[np.int32_t, ndim=1]
            Input indices.
        NUM_THREADS : int, default=1.
        """
        
        # Compute lengths to pass in
        cdef int rxlen = <int>rx.size
        cdef int shiftslen = <int>shifts.size
        
        # Allocate output
        cdef np.ndarray out = np.zeros((shifts.size, self.gxcfft.getFftlen()), dtype=np.float32)
        
        # Call the internal method
        self.gxcfft.xcorr(
            <Ipp32fc*>rx.data,
            rxlen,
            <int*>shifts.data,
            shiftslen,
            <Ipp32f*>out.data,
            NUM_THREADS
        )
        
        return out