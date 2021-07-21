from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from libc.stdint cimport uint8_t

# Declare the class with cdef
cdef extern from "viterbiDemodulator.cpp":
    pass

cdef extern from "viterbiDemodulator.h":
    ctypedef struct Ipp64fc:
        pass
    
    ctypedef double Ipp64f
    
    ctypedef unsigned char Ipp8u
    
    cdef cppclass ViterbiDemodulator:
        ViterbiDemodulator(Ipp64fc*, uint8_t, uint8_t*, uint8_t, uint8_t, Ipp64fc*, int, Ipp64f*, unsigned int) except +
        
        void preparePulseFilters()
        void freePulseFilters()
        
        string printAlphabet()
        string printValidTransitions()
        string printOmega()
        void printPathMetrics()
        void printPaths(int, int)
        void printBranchMetrics()
        string printOmegaVectors(int, int)
        string printPulses(int, int)
        
        int getWorkspaceIdx(int)
        void setUseThreading(bool)
        void setAllowedStartSymbolIndices(vector[Ipp8u] newAllowedIndices)
        
        void run(Ipp64fc*, int, int)
        void calcBranchMetrics(Ipp64fc*, int, int)
        
        void dumpOutput()
        