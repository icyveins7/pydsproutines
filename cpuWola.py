import numpy as np
import ctypes as ct
import os
os.environ['PATH'] = os.environ['PATH'] + os.pathsep + os.path.dirname(os.path.realpath(__file__))
os.add_dll_directory(os.path.join(os.environ['IPPROOT'], 'redist','intel64'))

def cpu_threaded_wola(y, f_tap, fftlen, Dec, NUM_THREADS=4):
    '''
    Has been edited to use the IPP FFT libraries. Now includes threads selection.
    Use this over all other methods. Commenting out the rest..
    '''
    if (len(f_tap) % fftlen != 0):
        print("Filter taps length must be factor multiple of fft length!")
        return 1
    # if (np.mod(fftlen,Dec)!=0):
    #     print("FFT length must be multiple of Decimation Factor!")
    #     return 1
    if (Dec * 2 != fftlen and Dec != fftlen):
        print(Dec)
        print(fftlen)
        print("PHASE CORRECTION ONLY IMPLEMENTED FOR DECIMATION = FFT LENGTH OR DECIMATION * 2 = FFT LENGTH!")
        return 1

    _libmc = np.ctypeslib.load_library('cpuWolaDll',loader_path=os.path.dirname(os.path.realpath(__file__)))
    array_1d_complexfloat = np.ctypeslib.ndpointer(dtype=np.complex64, ndim = 1, flags = 'CONTIGUOUS')
    array_1d_single = np.ctypeslib.ndpointer(dtype=np.float32, ndim = 1, flags = 'CONTIGUOUS')
    _libmc.cpuWola.restype = ct.c_int32
    _libmc.cpuWola.argtypes = [array_1d_complexfloat, array_1d_single, ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32, array_1d_complexfloat, ct.c_int32]

    siglen = len(y)
    out = np.empty(int(siglen/Dec * fftlen), dtype = np.complex64) # make the output

    retcode = _libmc.cpuWola(y, f_tap, fftlen, Dec, int(siglen/Dec), len(f_tap), out, NUM_THREADS) # run the dll function

    out = out.reshape((int(siglen/Dec),fftlen)) # reshape to channels in columns

    return out, retcode

# def cpu_threaded_wola_32fc(y, f_tap, fftlen, Dec):
#     if (len(f_tap) % fftlen != 0):
#         print("Filter taps length must be factor multiple of fft length!")
#         return 1
#     if (np.mod(fftlen,Dec)!=0):
#         print("FFT length must be multiple of Decimation Factor!")
#         return 1
# #    if (Dec != fftlen and Dec != fftlen*2):
# #        print(Dec)
# #        print(fftlen)
# #        print("PHASE CORRECTION ONLY IMPLEMENTED FOR DECIMATION = FFT LENGTH OR DECIMATION * 2 = FFT LENGTH!")
# #        return 1

#     _libmc = np.ctypeslib.load_library('cpuWolaDll_32fc', loader_path=os.path.dirname(os.path.realpath(__file__)))
#     array_1d_complex = np.ctypeslib.ndpointer(dtype=np.complex64, ndim = 1, flags = 'CONTIGUOUS')
#     array_1d_single = np.ctypeslib.ndpointer(dtype=np.float32, ndim = 1, flags = 'CONTIGUOUS')
#     _libmc.cpuWola.restype = ct.c_int32
#     _libmc.cpuWola.argtypes = [array_1d_complex, array_1d_single, ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32, array_1d_complex]

#     siglen = len(y)
#     out = np.empty(int(siglen/Dec * fftlen), dtype = np.complex64) # make the output

#     retcode = _libmc.cpuWola(y, f_tap, fftlen, Dec, int(siglen/Dec), len(f_tap), out) # run the dll function

#     out = out.reshape((int(siglen/Dec),fftlen)) # reshape to channels in columns

#     return out, retcode

# def cpu_threaded_wola_choosethreads(y, f_tap, fftlen, Dec, threads):
#     if (len(f_tap) % fftlen != 0):
#         print("Filter taps length must be factor multiple of fft length!")
#         return 1
#     if (np.mod(fftlen,Dec)!=0):
#         print("FFT length must be multiple of Decimation Factor!")
#         return 1
# #    if (Dec != fftlen and Dec != fftlen*2):
# #        print(Dec)
# #        print(fftlen)
# #        print("PHASE CORRECTION ONLY IMPLEMENTED FOR DECIMATION = FFT LENGTH OR DECIMATION * 2 = FFT LENGTH!")
# #        return 1

#     _libmc = np.ctypeslib.load_library('cpuWolaDll_choosethreads','.')
#     array_1d_complexdouble = np.ctypeslib.ndpointer(dtype=np.complex128, ndim = 1, flags = 'CONTIGUOUS')
#     array_1d_double = np.ctypeslib.ndpointer(dtype=np.float64, ndim = 1, flags = 'CONTIGUOUS')
#     _libmc.cpuWola.restype = ct.c_int32
#     _libmc.cpuWola.argtypes = [array_1d_complexdouble, array_1d_double, ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32, array_1d_complexdouble,ct.c_int32]

#     siglen = len(y)
#     out = np.empty(int(siglen/Dec * fftlen), dtype = np.complex128) # make the output

#     retcode = _libmc.cpuWola(y, f_tap, fftlen, Dec, int(siglen/Dec), len(f_tap), out, threads) # run the dll function

#     out = out.reshape((int(siglen/Dec),fftlen)) # reshape to channels in columns

#     return out, retcode
