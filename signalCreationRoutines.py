# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 15:18:01 2020

@author: Seo
"""

import numpy as np
import scipy as sp
import scipy.signal as sps
import time
import sympy

# from numba import jit, njit
# jit not used if not supported like in randint, or just slower..
# usually requires loops to be present for some benefit to be seen


# %%
def randBits(length, m):
    return np.random.randint(0, m, length, dtype=np.uint8)


def symsFromBits(bits, m, dtype=np.complex128):
    d = {
        2: np.array([1, -1], dtype=dtype),
        4: np.array([1, 1j, -1, -1j], dtype=dtype),
        8: np.array(
            [
                1,
                (1 + 1j) / np.sqrt(2),
                1j,
                (-1 + 1j) / np.sqrt(2),
                -1,
                (-1 - 1j) / np.sqrt(2),
                -1j,
                (1 - 1j) / np.sqrt(2),
            ],
            dtype=dtype,
        ),
    }
    return d[m][bits]

    # return np.exp(1j*2*np.pi*bits/m).astype(dtype)


def randPSKsyms(length: int, m: int, dtype=np.complex128):
    """
    Generates array of random m-ary PSK symbols.

    Parameters
    ----------
    length : int
        Length of array.
    m : int
        Order of PSK.
    dtype : numpy type, optional
        The default is np.complex128.

    Returns
    -------
    syms : np.ndarray
        Array of symbols.
    bits : np.ndarray
        Associated array of bits.

    """
    bits = randBits(length, m)
    return symsFromBits(bits, m, dtype), bits


def randnoise(length, bw_signal, chnBW, snr_inband_linear, sigPwr=1.0):
    """
    Generates an array of gaussian noise.

    Parameters
    ----------
    length : int
        Length of output array.
    bw_signal : float
        Bandwidth of the signal that this noise will be added to.
    chnBW : float
        Bandwidth of the noise i.e. usually the sampling rate.
    snr_inband_linear : float
        The desired SNR of the signal that this noise will be added to.
    sigPwr : float, optional
        The target signal power. This helps to scale the noise to the desired SNR.
        The default is 1.0.

    Returns
    -------
    noise : np.ndarray, np.complex128
        Noise array.
    """
    basicnoise = (
        (np.random.randn(length) + 1j * np.random.randn(length))
        / np.sqrt(2)
        * np.sqrt(sigPwr)
    )
    noise = (
        basicnoise * np.sqrt(1.0 / snr_inband_linear) *
        np.sqrt(chnBW / bw_signal)
    )  # pretty sure this is correct now..
    return noise


def addSigToNoise(
    signal: np.ndarray,
    noiseLen: int = None,
    sigStartIdx: int = 0,
    bw_signal: float = 1,
    chnBW: float = 1,
    snr_inband_linear: float = np.inf,
    sigPwr: float = 1.0,
    fshift: float = None,
):
    """Add signal into noisy background at particular index, with optional frequency shifting."""

    # if noise len not specified then use length of signal
    if noiseLen is None:
        noiseLen = len(signal)

    if snr_inband_linear is np.inf:
        print("Generating zeros for inf SNR..")
        noise = np.zeros(noiseLen, dtype=np.complex128)
    else:
        noise = randnoise(noiseLen, bw_signal, chnBW,
                          snr_inband_linear, sigPwr)
    aveNoisePwr = np.linalg.norm(noise) ** 2.0 / len(noise)
    print("Ave noise power = " + str(aveNoisePwr))
    aveSigPwr = np.linalg.norm(signal) ** 2.0 / len(signal)
    print("Ave sig power = " + str(aveSigPwr))
    expectedNoisePwr = (1.0 / snr_inband_linear) * chnBW / bw_signal
    print("Expected noise power = " + str(expectedNoisePwr))
    rx = np.zeros(noiseLen, dtype=np.complex128)
    rx[sigStartIdx: len(signal) + sigStartIdx] = signal
    rx = rx + noise

    if fshift is not None:
        tone = np.exp(1j * 2 * np.pi * fshift * np.arange(noiseLen) / chnBW)
        rx = rx * tone

        return noise, rx, tone
    else:
        return noise, rx


def addManySigToNoise(
    noiseLen,
    sigStartIdxList,
    signalList,
    bw_signal,
    chnBW,
    snr_inband_linearList,
    fshifts=None,
    sigStartTimeList=None,
):
    """
    Add many signals into noisy background at particular indices. with optional frequency shiftings.
    All signals are assumed to have signal power of unity i.e. sigPwr = 1.0 in the single generator function.

    This function will calculate a single noise array and scale the different signals appropriately to generate the
    necessary SNR differences.

    The noise array will be generated using the supplied scalar (not a list!) values of chnBW and bw_signal, so this
    implies that all signals added should in theory have the SAME BANDWIDTH as bw_signal, in order to achieve the
    desired relative SNR values (note, this is not SINR). This noise will use the first SNR in the list to generate
    the relative noise array.

    SNR supplied values cannot be infinity in this case (for obvious reasons).
    """
    # create standard noise with respect to the first SNR
    noise = randnoise(noiseLen, bw_signal, chnBW,
                      snr_inband_linearList[0], 1.0)

    # prepare the different time propagated versions of the noiseless signals
    numSigs = len(snr_inband_linearList)
    rx = np.zeros((numSigs, noiseLen), dtype=np.complex128)

    if (
        sigStartTimeList is None
    ):  # use the index version (faster if moving at sample level)
        for i in range(rx.shape[0]):
            rx[i][sigStartIdxList[i]: len(signalList[i]) + sigStartIdxList[i]] = (
                signalList[i]
                * np.sqrt(snr_inband_linearList[i] / snr_inband_linearList[0])
            )
    else:  # otherwise for subsample, move using the function
        # raise Exception("CURRENTLY DEBUGGING THIS MODE")
        for i in range(rx.shape[0]):
            rx[i][: len(signalList[i])] = signalList[i] * np.sqrt(
                snr_inband_linearList[i] / snr_inband_linearList[0]
            )  # set to 0

        ssTime = time.time()
        rx = propagateSignal(
            rx, sigStartTimeList, chnBW, freq=None, tone=None
        )  # then propagate required amount
        print("Subsample propagation took %fs." % (time.time() - ssTime))

    if fshifts is not None:

        tones = np.zeros((numSigs, noiseLen), dtype=np.complex128)

        for k in range(rx.shape[0]):
            tones[k] = np.exp(1j * 2 * np.pi * fshifts[k]
                              * np.arange(noiseLen) / chnBW)
            rx[k] = rx[k] * tones[k]

        rxfull = np.sum(rx, axis=0) + noise

        return noise, rxfull, tones

    else:
        rxfull = np.sum(rx, axis=0) + noise

        return noise, rxfull


def makeCPFSKsyms(bits, baud, m=2, h=0.5, up=8, phase=0.0):
    """
    Chose the same defaults as the comms toolbox in Matlab.
    Bits are expressed in 1s and 0s. Digital data (e.g. +/-1) is converted
    within the function itself.

    Note: this function assumes a rectangular pulse of amplitude 1/2T, length T.
    It has no 'pulse memory' of any other symbols. This is different from the idea
    of the accumulator, which accumulates the 'pulsed phase' value from all prior symbols.
    """
    T = 1.0 / baud
    fs = baud * up
    data = bits.astype(np.int8) * m - 1

    theta = np.zeros(len(bits) * up)

    # numpy version
    i_list = np.floor(np.arange(len(theta)) / up).astype(np.uint32)
    t_list = np.arange(len(theta)) / fs
    a_list = np.hstack(([0], np.cumsum(data)))[
        : len(data)]  # accumulator of phase
    a_list = np.repeat(a_list, up)

    theta = (
        (data[i_list] * np.pi * h * (t_list - i_list * T) / T)
        + np.pi * h * a_list
        + phase
    )

    sig = np.exp(1j * theta)

    return sig, fs, data


def makePulsedCPFSKsyms(bits, baud, g=np.ones(8) / 16, m=2, h=0.5, up=8, phase=0.0):
    """
    Uses the pulse shape g to create the signal.
    g is applied to the phase (in a convolutional way) before the actual symbols are created.
    This is in contrast to PSK where the pulse shape is applied to the symbols.
        With the default settings (pulse shape constant over one symbol), this should result in the same array
        as the non-pulsed function. The pulse shape is by default normalised to have integral 0.5 over the one symbol.

    In particular, the pulse shape function g is expected to already be scaled by the upsampling rate.
    In other words, the assumption within this function is that the calculations are done with a normalized sampling rate.

    Note: this function will return the full convolution. It is up to the user to
    define where the end or start index is depending on the pulse shape centre, and the
    corresponding delay introduced. In the default rect-pulse, there is no clear 'delay', so the
    output can start from index 0 (as it is in the non pulsed function), and the ending len(g) indices
    can be trimmed off.
    """
    T = 1.0 / baud  # symbol period
    fs = baud * up  # sample period
    data = bits.astype(np.int8) * m - 1

    theta = np.zeros(len(bits) * up + 1)

    # first create the upsampled version of the data (pad zeros between elements)
    theta[1::up] = data  # we need the zero at the start (before accumulation)

    # then convolve it with the pulse
    c = sps.convolve(theta, g)

    # and now accumulate the phase (from a starting idx)
    # note, this assumes a normalized sampling rate (fs = 1 Hz)!
    # likewise this assumes that the pulse shape has accounted for this value
    cs = np.cumsum(c)

    # scale by 2 pi h and initial phase
    css = cs * 2 * np.pi * h + phase

    sig = np.exp(1j * css)

    return sig, fs, data, css


def propagateSignal(sig, time, fs, freq=None, tone=None):
    # to handle single scalar time shift
    if not isinstance(time, np.ndarray):
        time = np.array([time])

    # to handle 1-D input
    if sig.ndim == 1:
        # automatic 2-d row vector detection using -1
        sig = sig.reshape((1, -1))

    # generate a tone if no tone is passed in and a freqshift is desired
    if freq is not None and tone is None:
        # print('Generating tone for freq shift..')
        tone = np.exp(1j * 2 * np.pi * freq * np.arange(sig.shape[1]) / fs)

    # propagate the signal in time
    sigfft = np.fft.fft(sig)  # this automatically ffts each row
    sigFreq = makeFreq(sig.shape[1], fs).reshape(
        (1, sig.shape[1])
    )  # construct 2-d, row vector
    mat = np.exp(
        1j * 2 * np.pi * sigFreq * -time.reshape((len(time), 1))
    )  # construct 2d matrix for each row having its own time shift
    preifft = mat * sigfft
    result = np.fft.ifft(preifft)

    # no freq shift, just return
    if tone is None:
        return result
    # otherwise return the freqshifted version with the tone
    else:
        # print('Returning shifted signal + tone used.')
        return result * tone, tone


def propagateSignalExact(sig, tau, fs, f_c=0.0):
    # take the fft of signal
    fftsig = np.fft.fft(sig)

    # pre-allocate
    result = np.zeros(len(sig), dtype=sig.dtype)

    # loop over n values of sig, continuous version
    N = len(sig)
    for n in np.arange(N):
        ntau = n / fs - tau[n]

        pseudotone = np.exp(1j * 2 * np.pi * ntau * makeFreq(N, fs))

        p = 1.0 / N * pseudotone * fftsig

        result[n] = np.sum(p)

    # scale by the carrier frequency to induce the doppler
    carrier = np.exp(-1j * 2 * np.pi * f_c * tau)
    result = result * carrier

    return result


def padZeros_fftfactors(sig, minpad, fftprimeMax=7):
    lensig = len(sig)

    totallen = lensig + minpad - 1
    found = False

    while not found:
        totallen = totallen + 1

        d = sympy.ntheory.factorint(totallen)

        found = True  # start by assuming this is correct
        for i in d.keys():
            if i > fftprimeMax:  # if any of them exceed then this test fails
                found = False

    # after the loop just return everything
    padlen = totallen - lensig
    out = np.pad(sig, [0, padlen])

    return out, padlen, d


# @jit(nopython=True)
def makeFreq(length, fs):
    freq = np.zeros(length)
    for i in range(length):
        freq[i] = i / length * fs
        if freq[i] >= fs / 2:
            freq[i] = freq[i] - fs
    return freq


def timeSliceSignal(x: np.ndarray, tstart: float, tstop: float, fs: float):
    """
    Convenience function to extract a time slice of a signal.
    This is easier to use when you have already plotted the signal with time x-axis
    rather than sample x-axis.
    """
    return x[int(tstart * fs): int(tstop * fs)]


def freqshiftSignal(x: np.ndarray, freq: float, fs: float = 1.0) -> np.ndarray:
    """
    Convenience function apply frequency shift to a signal.

    Parameters
    ----------
    x : np.ndarray
        Input signal
    freq : float
        Frequency shift to apply, in Hz.
    fs : float, optional
        Sampling rate of input, by default 1.0.

    Returns
    -------
    shifted : np.ndarray
        x * np.exp(1j*2*np.pi*freq*t)
    """
    tone = np.exp(1j * 2 * np.pi * freq * np.arange(x.size) / fs)
    shifted = tone * x
    return shifted


# %%
try:
    import cupy as cp
    import os

    # Raw kernel for tone creation
    addPhaseKernel = cp.RawKernel(
        r""" 
    extern "C" __global__
    void addPhase(
        float *phase,
        int len,
        double freq,
        double tstart,
        double tstep)
    {
         int tidx = blockIdx.x * blockDim.x + threadIdx.x;
        
         // constant
         const double TWO_PI_F = 6.283185307179586 * freq;
         
         // grid stride
         for (int i = tidx; i < len; i = i + gridDim.x * blockDim.x)
         {
             // perform the calculation in double (required) but cast to float
             phase[i] = (float)fma(TWO_PI_F, fma((double)i, tstep, tstart), (double)phase[i]);
         }
     
    }
    """,
        """addPhase""",
    )

    def cupyAddTonePhase(phase: cp.ndarray, freq: float, tstart: float, tstep: float):
        """
        Generates the phase values of a tone i.e. 2 pi f t, and adds it to the given array.
        Is generally ~5-6 times faster than running the equivalent
        phase = phase + 2 * cp.pi * freq * cp.arange(phase.size) * T

        Parameters
        ----------
        phase : cp.ndarray
            Input/output array. Is added to in-place.
        freq : float
            Frequency value.
        tstart : float
            Time value of first sample.
        tstep : float
            Time step per sample.

        Raises
        ------
        TypeError
            Expects phase to be 32-bit.

        """
        if phase.dtype != cp.float32:
            raise TypeError("Phase is expected to be 32-bit float.")

        THREADS_PER_BLOCK = 256
        NUM_BLOCKS = phase.size // THREADS_PER_BLOCK + 1

        addPhaseKernel(
            (NUM_BLOCKS,),
            (THREADS_PER_BLOCK,),
            (phase, phase.size, freq, tstart, tstep),
        )

    # Raw kernels for tone creation
    with open(
        os.path.join(os.path.dirname(__file__),
                     "custom_kernels", "genTones.cu"), "r"
    ) as fid:
        module = cp.RawModule(code=fid.read())
        genTonesDirect_64fKernel = module.get_function("genTonesDirect_64f")
        genTonesScaling_64fKernel = module.get_function("genTonesScaling_64f")
        genTonesDirect_32fKernel = module.get_function("genTonesDirect_32f")
        genTonesScaling_32fKernel = module.get_function("genTonesScaling_32f")

    def cupyGenTonesDirect(
        f0: float,
        fstep: float,
        numFreqs: int,
        length: int,
        dtype: type = np.complex128,
        THREADS_PER_BLOCK: int = 128,
    ):
        """This kernel is almost 1 order slower than the scaling kernel according to nvprof."""
        # Simple checks
        if abs(f0) > 1.0 or f0 + (numFreqs - 1) * fstep >= 1.0:
            raise ValueError("Frequencies should be normalised.")

        NUM_BLOCKS = length // THREADS_PER_BLOCK + 1

        out = cp.empty((numFreqs, length), dtype=dtype)

        if dtype == np.complex128:
            genTonesDirect_64fKernel(
                (NUM_BLOCKS,), (THREADS_PER_BLOCK,
                                ), (f0, fstep, numFreqs, length, out)
            )

        elif dtype == np.complex64:
            genTonesDirect_32fKernel(
                (NUM_BLOCKS,), (THREADS_PER_BLOCK,
                                ), (f0, fstep, numFreqs, length, out)
            )

        else:
            raise TypeError("dtype must be either complex128 or complex64")

        return out

    def cupyGenTonesScaling(
        f0: float,
        fstep: float,
        numFreqs: int,
        length: int,
        dtype: type = np.complex128,
        THREADS_PER_BLOCK: int = 128,
    ):
        """This may appear to be slower when timing within python, but the kernel is significantly faster (~10x via nvprof)."""
        NUM_BLOCKS = length // THREADS_PER_BLOCK + 1

        out = cp.empty((numFreqs, length), dtype=dtype)

        if dtype == np.complex128:
            genTonesScaling_64fKernel(
                (NUM_BLOCKS,), (THREADS_PER_BLOCK,
                                ), (f0, fstep, numFreqs, length, out)
            )
        elif dtype == np.complex64:
            genTonesScaling_32fKernel(
                (NUM_BLOCKS,), (THREADS_PER_BLOCK,
                                ), (f0, fstep, numFreqs, length, out)
            )
        else:
            raise TypeError("dtype must be either complex128 or complex64")

        return out

except ModuleNotFoundError as e:
    print("Cupy not found. Ignoring cupy imports.")


# %%
if __name__ == "__main__":
    from verifyRoutines import *
    from timingRoutines import *

    timer = Timer()

    try:
        import cupy as cp

        # Test params
        f0 = 0.0
        numFreqs = 1000
        fstep = 1 / numFreqs
        length = numFreqs

        # Actual test vs standard cupy calls
        timer.start()
        customtones = cupyGenTonesDirect(f0, fstep, numFreqs, length)
        timer.evt("direct kernel")
        customtones_scaling = cupyGenTonesScaling(f0, fstep, numFreqs, length)
        timer.evt("scaling kernel")
        tones = cp.exp(
            1j
            * 2
            * cp.pi
            * cp.arange(f0, 1.0, fstep).reshape((-1, 1))
            * cp.arange(length)
        )
        timer.evt("cupy")

        # Test 32f version
        customtone32f = cupyGenTonesDirect(
            f0, fstep, numFreqs, length, np.complex64)
        timer.evt("direct kernel 32f")
        customtone32f_scaling = cupyGenTonesScaling(
            f0, fstep, numFreqs, length, np.complex64
        )
        timer.evt("scaling kernel 32f")

        timer.end()
        print("Comparing naive tones kernel with cupy")
        compareValues(customtones.get().flatten(), tones.get().flatten())
        print("Comparing scaling tones kernel with cupy")
        compareValues(customtones_scaling.get().flatten(),
                      tones.get().flatten())
        print("Comparing 32f tones kernel with cupy")
        compareValues(customtone32f.get().flatten(), tones.get().flatten())
        print("Comparing 32f tones scaling kernel with cupy")
        compareValues(customtone32f_scaling.get().flatten(),
                      tones.get().flatten())

    except ModuleNotFoundError as e:
        print("Skipping cupy-related tests.")
