import numpy as np
import ctypes as ct
import os
os.environ['PATH'] = os.environ['PATH'] + os.pathsep + os.path.dirname(os.path.realpath(__file__))
if os.name == 'nt':
    os.add_dll_directory(os.path.join(os.environ['IPPROOT'], 'redist','intel64'))
elif os.name == 'posix':
    os.environ['PATH'] = os.environ['PATH'] + os.pathsep + os.path.join(os.environ['IPPROOT'], 'lib', 'intel64')

def cpuTone(length: int, freq: float, fs: int, phase: float = 0):
    '''
    IPP wrapper for ippsTone.
    '''

    _libmc = np.ctypeslib.load_library('cpuToneDll',loader_path=os.path.dirname(os.path.realpath(__file__)))
    array_1d_complexdouble = np.ctypeslib.ndpointer(dtype=np.complex128, ndim = 1, flags = 'CONTIGUOUS')
    _libmc.cpuTone.restype = ct.c_int32
    _libmc.cpuTone.argtypes = [ct.c_int32, ct.c_double, ct.c_double, ct.c_double, array_1d_complexdouble]

    out = np.empty(int(length), dtype = np.complex128) # make the output

    retcode = _libmc.cpuTone(length, freq, float(fs), phase, out)

    return out, retcode
