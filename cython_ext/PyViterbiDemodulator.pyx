# distutils: language = c++

import numpy as np
cimport numpy as np
from viterbiDemodulator cimport ViterbiDemodulator, Ipp64fc, Ipp64f, Ipp8u, uint8_t

from libcpp.string cimport string

cdef class PyViterbiDemodulator:
    cdef ViterbiDemodulator* vd
    
    def __cinit__(self, np.ndarray[np.complex128_t, ndim=1] in_alphabet, unsigned int alphabetLen,
            		np.ndarray[np.uint8_t, ndim=2] in_preTransitions, unsigned int preTransitionsLen,
            		unsigned int in_numSrc,
            		np.ndarray[np.complex128_t, ndim=1] in_pulses, int in_pulselen, 
            		np.ndarray[np.float64_t, ndim=1] in_omegas, 
            		unsigned int in_up):
        
        print("Attempting to init..")
        
        contiguousPretransitions = np.ascontiguousarray(in_preTransitions, dtype=np.uint8)
        # contiguousPulses = np.ascontiguousarray(in_pulses, dtype=np.complex128) # this doesn't work?
        # print(contiguousPulses)
        
        self.vd = new ViterbiDemodulator(<Ipp64fc*>in_alphabet.data, <uint8_t>alphabetLen,
                                         <bytes>contiguousPretransitions.data.tobytes(), <uint8_t>preTransitionsLen, # DO NOT CHANGE FROM <BYTES>, NOR THE .tobytes() !!!
                                         <uint8_t>in_numSrc,
                                         <Ipp64fc*>in_pulses.data, in_pulselen,
                                         <Ipp64f*>in_omegas.data,
                                         in_up)
        
        self.printAlphabet()
        self.printValidTransitions()
        self.printOmega()
        
    def __dealloc__(self):
        del self.vd
        
    def printAlphabet(self):
        print(self.vd.printAlphabet().decode('utf-8'))
        
    def printValidTransitions(self):
        print(self.vd.printValidTransitions().decode('utf-8'))
        
    def printOmega(self):
        print(self.vd.printOmega().decode('utf-8'))
        
    def printPathMetrics(self):
        self.vd.printPathMetrics()
        
    def printPaths(self, n, s=0):
        self.vd.printPaths(n,s)
        
    def printBranchMetrics(self):
        self.vd.printBranchMetrics()
        
    def printOmegaVectors(self, s, e):
        print(self.vd.printOmegaVectors(s,e).decode('utf-8'))
        
    def printPulses(self, s, e):
        print(self.vd.printPulses(s,e).decode('utf-8'))
        
    def getWorkspaceIdx(self, s):
        return self.vd.getWorkspaceIdx(s)
    
    def setAllowedStartSymbolIndices(self):
        return None # not yet implemented
    
    def setUseThreading(self, i):
        self.vd.setUseThreading(i)
        
    def run(self, np.ndarray[np.complex128_t, ndim=1] y, ylength, pathlen):
        self.vd.run(<Ipp64fc*>y.data, ylength, pathlen)
        
    def calcBranchMetrics(self, np.ndarray[np.complex128_t, ndim=1] y, n, pathlen):
        self.vd.run(<Ipp64fc*>y.data, n, pathlen)
        
    def dumpOutput(self):
        self.vd.dumpOutput()
    