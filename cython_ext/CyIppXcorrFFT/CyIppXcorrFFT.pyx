# distutils: language = c++
import numpy as np
cimport numpy as np
from IppXcorrFFT cimport IppXcorrFFT_32fc, Ipp32fc, bool

cdef class CyIppXcorrFFT:
    """
    Cythonised version of IppXcorrFFT.
    """

    cdef IppXcorrFFT_32fc* xcfft
    
    def __cinit__(self,
                  np.ndarray[np.complex64_t, ndim=1] cutout,
                  int num_threads=1,
                  bool autoConj=True):
        """
        Instantiates a CyIppXcorrFFT object.
        """
        
        self.xcfft = new IppXcorrFFT_32fc(
            <Ipp32fc*> cutout.data, <int> cutout.size, num_threads, autoConj
        )

    def __dealloc__(self):
        del self.xcfft
        
    # Main computation method
    def xcorr(self,
              np.ndarray[np.complex64_t, ndim=1] rx,
              int startIdx,
              int endIdx,
              int step):
        
        # allocate numpy array output
        cdef length = len(np.arange(startIdx, endIdx, step))
        cdef np.ndarray productpeaks = np.zeros(length, dtype=np.float32)
        cdef np.ndarray freqlistinds = np.zeros(length, dtype=np.int32)

        # Call the internal method
        self.xcfft.xcorr_array(
            <Ipp32fc*>rx.data,
            <int>rx.size,
            startIdx,
            endIdx,
            step,
            <float*>productpeaks.data,
            <int*>freqlistinds.data,
            length
        )

        return productpeaks, freqlistinds

    def results(self):
        return self.xcfft.m_productpeaks, self.xcfft.m_freqlistinds
        
