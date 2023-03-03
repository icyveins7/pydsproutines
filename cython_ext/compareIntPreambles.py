import numpy as np
import ctypes as ct
import os

def compareIntPreambles(
    preamble: np.ndarray,
    x: np.ndarray,
    m: int,
    searchStart: int=0,
    searchEnd: int=None
):

    #### Input argument checks
    # Default searchEnd value
    if searchEnd is None:
        searchEnd = x.size - preamble.size
    elif searchEnd > x.size - preamble.size:
        raise ValueError("searchEnd must fit the preamble length")
    
    if preamble.dtype != np.uint8:
        raise TypeError("preamble should be uint8.")
    if x.dtype != np.uint8:
        raise TypeError("x should be uint8.")
    if searchStart < 0 or searchStart >= searchEnd:
        raise ValueError("searchStart should be >=0 and before searchEnd")
    
    # Load DLL
    _libmc = np.ctypeslib.load_library('compareIntPreambles',loader_path=os.path.dirname(os.path.realpath(__file__)))

    array_2d_uint32 = np.ctypeslib.ndpointer(dtype=np.uint32, ndim=2, flags='CONTIGUOUS')
    array_1d_uint8 = np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags='CONTIGUOUS')

    _libmc.compareIntPreambles.restype = ct.c_int32
    _libmc.compareIntPreambles.argtypes = [
        array_1d_uint8,
        ct.c_int32,
        array_1d_uint8,
        ct.c_int32,
        ct.c_int32,
        ct.c_int32,
        ct.c_int32,
        array_2d_uint32]

    # Allocate output
    matches = np.zeros((searchEnd-searchStart, m), dtype=np.uint32)
    
    # Run the dll
    retcode = _libmc.compareIntPreambles(preamble, preamble.size, x, x.size, searchStart, searchEnd, m, matches)
    if retcode != 0:
        raise RuntimeError("DLL returned an error.")

    return matches

