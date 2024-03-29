# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 16:21:35 2020

@author: Seo
"""

import numpy as np
from timingRoutines import Timer

# from numba import njit, jit
from xcorrRoutines import *
import warnings
import matplotlib.pyplot as plt
from cython_ext.compareIntPreambles import compareIntPreambles

try:
    import cupy as cp

    def cupyDemodulateCP2FSK(syms: cp.ndarray, h: float, up: int):
        m = cp.array([[-1], [+1]])

        tones = cp.exp(1j * np.pi * h * cp.arange(up) / up * m)

        numSyms = int(np.floor(len(syms) / up))
        bitCost = cp.zeros((2, numSyms))
        demodBits = cp.zeros(numSyms, dtype=np.uint8)

        for i in range(numSyms):
            symbol = syms[i * up : (i + 1) * up]

            for k in range(2):
                bitCost[k, i] = cp.abs(cp.vdot(symbol, tones[k]))

            demodBits[i] = cp.argmax(bitCost[:, i])

        return demodBits, bitCost, tones

except:
    print("Cupy not found.")


# %% Generic simple demodulators
class SimpleDemodulatorPSK:
    """
    Generic demodulator implementation for BPSK/QPSK/8PSK.
    This uses a dot product method to detect which symbol in the constellation is present.

    The default constellation-bit mappings are provided as gray-mapped bits,
    but this can be changed.
    """

    # These default psk constellations are provided only for the basic class.
    # The specialised classes use their own constellations which are optimised for demodulation.
    pskdicts = {  # This is a monotonically increasing index for each increase in angle
        2: np.array([1.0, -1.0], dtype=np.complex128),
        4: np.array([1.0, 1.0j, -1.0, -1.0j], dtype=np.complex128),
        8: np.array(
            [
                1.0,
                np.sqrt(2) / 2 * (1 + 1j),
                1.0j,
                np.sqrt(2) / 2 * (-1 + 1j),
                -1.0,
                np.sqrt(2) / 2 * (-1 - 1j),
                -1.0j,
                np.sqrt(2) / 2 * (1 - 1j),
            ],
            dtype=np.complex128,
        ),
    }
    # This is a specific bit mapping for each corresponding index i.e. each angle, in increasing order
    # E.g. QPSK/8PSK are gray mapped.
    pskbitmaps = {
        2: np.array([0b1, 0b0], dtype=np.uint8),
        4: np.array([0b11, 0b01, 0b00, 0b10], dtype=np.uint8),
        8: np.array(
            [0b000, 0b001, 0b011, 0b010, 0b110, 0b111, 0b101, 0b100], dtype=np.uint8
        ),
    }

    def __init__(
        self, m: int, bitmap: np.ndarray = None, cluster_threshold: float = 0.1
    ):
        self.m = m
        self.const = self.pskdicts[self.m]
        self.normVecs = self.const.view(np.float64).reshape((-1, 2))
        self.bitmap = self.pskbitmaps[self.m] if bitmap is None else bitmap
        self.cluster_threshold = cluster_threshold

        # Interrim output
        self.xeo = None  # Selected eye-opening resample points
        self.xeo_i = None  # Index of eye-opening
        self.eo_metric = None  # Metrics of eye-opening
        self.reimc = None  # Phase-locked to constellation (complex array)
        self.svd_metric = None  # SVD metric for phase lock
        self.angleCorrection = None  # Angle correction used in phase lock
        self.syms = None  # Output mapping to each symbol (0 to M-1)
        self.matches = None  # Output from amble rotation search

    def getEyeOpening(self, x: np.ndarray, osr: int, abs_x: np.ndarray = None):
        if abs_x is None:
            abs_x = np.abs(
                x
            )  # Provide option for pre-computed (often used elsewhere anyway)
        x_rs_abs = abs_x.reshape((-1, osr))
        self.eo_metric = np.mean(x_rs_abs, axis=0)
        i = np.argmax(self.eo_metric)
        x_rs = x.reshape((-1, osr))
        return x_rs[:, i], i

    def mapSyms(self, reimc: np.ndarray):
        """
        Maps symbols to values from 0 to m-1. Note that this may not correspond to the
        bit values desired e.g. gray mapping. In such scenarios, the bitmap should be amended.

        This method does not need to be called directly; it is called as part of demod().

        See symsToBits() for actual bit mapping.

        Parameters
        ----------
        reimc : np.ndarray
            Correct eye-opening, frequency corrected and phase-locked complex-valued input.

        Returns
        -------
        syms : np.ndarray
            Output array corresponding to the symbol values 0 to m-1.

        """
        reimcr = reimc.view(np.float32).reshape((-1, 2)).T
        constmetric = self.normVecs @ reimcr
        # Pick the arg max for each column
        syms = np.argmax(constmetric, axis=0).astype(np.uint8)

        return syms

    def lockPhase(self, reim: np.ndarray):
        # Power into BPSK
        powerup = self.m // 2
        reimp = reim**powerup

        # Form the square product
        reimpr = reimp.view(np.float32).reshape((-1, 2)).T
        reimsq = reimpr @ reimpr.T

        # SVD
        u, s, vh = np.linalg.svd(reimsq)  # Don't need vh technically
        # Check the svd metrics
        svd_metric = (
            s[-1] / s[:-1]
        )  # Deal with this later when there is residual frequency
        if np.any(svd_metric > self.cluster_threshold):
            warnings.warn(
                "Constellation not well clustered. There may be residual frequency shifts."
            )
        # Angle correction
        angleCorrection = np.arctan2(u[1, 0], u[0, 0])
        reimc = self.correctPhase(reim, -angleCorrection / powerup)

        return reimc, svd_metric, angleCorrection

    def correctPhase(self, reim: np.ndarray, phase: float):
        return reim * np.exp(1j * phase)

    def demod(
        self, x: np.ndarray, osr: int, abs_x: np.ndarray = None, verb: bool = True
    ):
        if x.dtype != np.complex64:
            raise TypeError("Input array must be complex64.")

        timer = Timer()
        timer.start()

        # Get eye-opening first
        xeo, xeo_i = self.getEyeOpening(x, osr, abs_x)
        timer.evt("Eye-opening")

        # Correct the global phase first
        reim = np.ascontiguousarray(xeo)
        self.reimc, self.svd_metric, self.angleCorrection = self.lockPhase(reim)
        timer.evt("lockPhase")

        # Generic method: dot product with the normalised vectors
        self.syms = self.mapSyms(self.reimc)
        timer.evt("mapSyms")

        if verb:
            timer.rpt()

        return self.syms

    def ambleRotate(
        self, amble: np.ndarray, search: np.ndarray = None, syms: np.ndarray = None
    ):
        if syms is None:
            syms = self.syms

        if search is None:
            search = np.arange(syms.size - amble.size + 1)

        # Naive loop
        length = amble.size

        # Custom dll
        self.matches = compareIntPreambles(
            amble, syms, self.m, search[0], search[-1] + 1
        )

        s, rotation = argmax2d(self.matches)
        sample = search[s]  # Remember to reference the searched indices
        rotatedsyms = (syms + rotation) % self.m  # We rotate and return this!
        bestMatches = self.matches[s, rotation]

        return rotatedsyms, sample, rotation, bestMatches

    @staticmethod
    # @njit('uint32[:,:](uint8[:], int32[:], intc, uint8[:], intc)', cache=True, nogil=True)
    def _ambleSearch(m_amble, search, m, syms, length):
        matches = np.zeros((search.size, m), dtype=np.uint32)
        for i in np.arange(search.size):  # Use np.arange instead of range
            mi = search[i]
            diff = np.mod((m_amble - syms[mi : mi + length]), m)

            # One-pass loop
            for k in np.arange(diff.size):
                matches[i, diff[k]] += 1

        return matches

    # @staticmethod
    # @njit(cache=True, nogil=True) # not well tested yet
    # def _ambleSearchv2(m_amble, search, m, syms, length):
    #     matches = np.zeros((search.size, m), dtype=np.uint32)
    #     for i in np.arange(search.size): # Use np.arange instead of range
    #         mi = search[i]
    #         diff = np.bitwise_xor(amble, syms[mi:mi+length])
    #         # One-pass loop
    #         for k in np.arange(diff.size):
    #             matches[i, -1-diff[k]] += 1

    #     return matches

    def symsToBits(self, syms: np.ndarray = None, phaseSymShift: int = 0):
        """
        Maps each symbol (integer array denoting the angle) to its own bit sequence,
        as specified by the bitmap.

        Parameters
        ----------
        syms : np.ndarray, uint8, optional
            Input symbol sequence. The default is None, which will use the last internally saved
            syms array output.

        phaseSymShift : int
            Number of symbols to rotate the bit mapping by.
            Example: m = 4.
                Current bitmap is [3,1,0,2].
                Rotating by 2 symbols equates to a phase shift of pi
                (or equivalently, phase shift of syms by -pi).

        Returns
        -------
        bits : np.ndarray
            Bit sequence stored as individual bytes i.e. length of this array = length of syms.

        """
        if syms is None:
            syms = self.syms

        return np.roll(self.bitmap, phaseSymShift)[syms]

    def unpackToBinaryBytes(self, packed: np.ndarray):
        """
        Turns an integer valued output from mapSyms()/demod()/symsToBits() into
        a binary-valued array with each row corresponding to the binary value
        of the integer.

        Specifically, that means that each bit now occupies one byte in memory,
        hence the name of the method. Contrast this with the packBinaryBytesToBits()
        method which tends to follow.

        Example:
            m = 4.
            Input array [0,1,2,3].
            Output is [[0,0],
                       [0,1],
                       [1,0],
                       [1,1]].

        Parameters
        ----------
        packed : np.ndarray
            Integer valued array.

        Returns
        -------
        unpacked : np.ndarray
            Matrix of N x k binary values, where N is the original length of 'packed',
            and k is the number of bits used to represent each value of 'packed',
            given by log2(m).

        """

        bitsPerVal = int(np.log2(self.m))
        # Unpack as usual
        unpacked = np.unpackbits(packed).reshape((-1, 8))
        # Slice the ending bits (default is big-endian)
        unpacked = unpacked[:, -bitsPerVal:]

        return unpacked

    def packBinaryBytesToBits(self, unpacked: np.ndarray):
        """
        This is a simple wrapper around numpy's packbits().
        In this context, it takes the unpacked matrix from unpackToBinaryBytes()
        and then compresses it to occupy the minimum requirement of bytes storage.

        Example:
            Input (QPSK) array [[0,0],
                                [0,1],
                                [1,0],
                                [1,1]].
            This is compressed to a single byte corresponding to
            [0,0,0,1,1,0,1,1], which is then returned as array([27]).

        Parameters
        ----------
        unpacked : np.ndarray
            Input unpacked bits, usually from unpackToBinaryBytes().

        Returns
        -------
        packed : np.ndarray
            Packed bits storage of the input.
        """
        return np.packbits(unpacked.reshape(-1))

    def findPlainText(self, syms: np.ndarray = None, phaseSymShift: int = 0):
        """
        For fixed symbols input and phaseSymShift mapping,
        attempts to find an appropriate number of symbols to skip to maximise
        the number of readable characters in UTF-8 encoding.

        UTF-8 characters lie within 0x21 to 0x7E. For blind demodulation,
        it may not be clear where the start of a byte is.

        E.g. QPSK has 2 bits per symbol.
        Hence there are 4 possible 'alignments' to read the start of a byte.

        This method will attempt to search the possible alignments and return the best one.

        Parameters
        ----------
        syms : np.ndarray
            Input array, usually from demod() output. Defaults to None,
            which uses the internally saved output from the last demod().
        phaseSymShift : int
            Bitmap rotation, similar to symsToBits(). Defaults to 0.

        Returns
        -------
        iSkip : int
            The maximised alignment. The text can be read by then using
            syms[iSkip:].
        utf8chars : np.ndarray
            Number of readable characters for the particular alignment.

        """
        if syms is None:
            syms = self.syms

        # BPSK: 8 symbols
        # QPSK: 4 symbols
        # 8PSK: 24 symbols (due to 3x8)
        symbolSkips = np.arange(np.lcm(self.m, 8), dtype=np.uint32)

        # Search for the best one
        utf8chars = np.zeros(symbolSkips.size, dtype=np.uint32)
        for i, symbolSkip in enumerate(symbolSkips):
            mapped = self.symsToBits(syms[symbolSkip:], phaseSymShift)
            packedbytes = self.packBinaryBytesToBits(self.unpackToBinaryBytes(mapped))
            # Characters in UTF-8 start at 0x21, end at 0x7E
            utf8chars[i] = np.intersect1d(
                np.argwhere(packedbytes >= 0x21).reshape(-1),
                np.argwhere(packedbytes <= 0x7E).reshape(-1),
            ).size

        # Maximise the skip with most readable characters
        iSkip = np.argmax(utf8chars)

        return iSkip, utf8chars

    @staticmethod
    def detect_B_or_Q(reim: np.ndarray, threshold: float = 0.5):
        """
        Detects if a complex array is BPSK or QPSK, by comparing
        the eigenvalues of an SVD decomposition.

        BPSK eigenvalues are likely to be very different, whereas
        QPSK eigenvalues are likely to be roughly equal.
        The ratio of the eigenvalues is compared to the threshold;
        QPSK is thus likely to be near 1.0 and BPSK is likely to be near 0.0.

        Parameters
        ----------
        reim : np.ndarray
            Input complex array(s) at baudrate.
            For 2-D arrays, each row is processed.

            This is usually the output after
            eye opening detection. No phase rotation is necessary.

        threshold : float, optional
            Threshold for BPSK/QPSK detection. The default is 0.5.

        Returns
        -------
        m : np.ndarray
            2 if BPSK, 4 if QPSK, for each row of the input array.
        """
        # Check type
        if reim.dtype != np.complex64 and reim.dtype != np.complex128:
            raise TypeError("Input array must be complex.")

        # Check shape
        if reim.ndim == 1:
            reim = reim.reshape((1, -1))  # Reshape to 1xN

        # Allocate output
        m = np.zeros(reim.shape[0], dtype=np.uint8)
        yl = np.zeros(reim.shape[0])

        # Iterate over every row
        for i, row in enumerate(reim):
            # Collapse to Nx2 real
            x = row.astype(np.complex128).view(np.float64).reshape((-1, 2))

            # Perform the svd on the self-dot-product
            _, s, _ = np.linalg.svd(x.T @ x)  # 2x2

            # Determine if B or Q based on eigenvalues
            y = s[1] / s[0]
            yl[i] = y
            if y < threshold:
                m[i] = 2
            else:
                m[i] = 4

        return m, yl


###############
class SimpleDemodulatorBPSK(SimpleDemodulatorPSK):
    """
    Faster demodulator implementation specifically for BPSK.
    """

    def __init__(self, bitmap: np.ndarray = None, cluster_threshold: float = 0.1):
        super().__init__(2, bitmap, cluster_threshold)

    @staticmethod
    def mapSyms(reimc: np.ndarray):
        # Simply get the real
        re = np.real(reimc)

        # And check sign
        syms = (re < 0).astype(np.uint8)

        return syms


###############
class SimpleDemodulatorQPSK(SimpleDemodulatorPSK):
    """
    Faster demodulator implementation specifically for QPSK.
    """

    gray4 = np.array([[2, 1], [3, 0]], dtype=np.uint8)

    def __init__(self, bitmap: np.ndarray = None, cluster_threshold: float = 0.1):
        super().__init__(4, bitmap, cluster_threshold)

        # self.gray4 = np.zeros((2,2), dtype=np.uint8)
        # self.gray4[1,1] = 0
        # self.gray4[0,1] = 1
        # self.gray4[0,0] = 2
        # self.gray4[1,0] = 3
        # This is X,Y > 0 gray encoded

    @staticmethod
    def mapSyms(reimc: np.ndarray):
        if reimc.dtype != np.complex64:
            raise TypeError("Input array must be complex64.")

        # Reshape
        reimd = reimc.view(np.float32).reshape((-1, 2))

        # # Compute comparators
        # xp = (reimd[:,0] > 0).astype(np.uint8)
        # yp = (reimd[:,1] > 0).astype(np.uint8)

        # # Now map
        # idx = np.vstack((xp,yp))
        # # Convert to constellation integers
        # syms = self.gray4[tuple(idx)]

        # New one-liner, prevents multiple comparator calls hence faster?
        syms = SimpleDemodulatorQPSK.gray4[tuple((reimd > 0).T.astype(np.uint8))]

        return syms

    def correctPhase(self, reim: np.ndarray, phase: float):
        # For gray-coding comparators, we move to the box
        return reim * np.exp(1j * (phase + np.pi / 4))


################
class SimpleDemodulator8PSK(SimpleDemodulatorPSK):
    """
    Faster demodulator implementation specifically for 8PSK.
    """

    def __init__(self, bitmap: np.ndarray = None, cluster_threshold: float = 0.1):
        super().__init__(8, bitmap, cluster_threshold)

        # For the custom constellation, we don't map to a number but rather to the N-D index,
        # mirroring the actual bits.
        self.map8 = np.zeros((2, 2, 2), dtype=np.uint8)
        self.map8[1, 1, 1] = 0
        self.map8[0, 1, 1] = 1
        self.map8[1, 0, 1] = 2
        self.map8[0, 0, 1] = 3
        self.map8[1, 1, 0] = 4
        self.map8[0, 0, 0] = 5
        self.map8[1, 0, 0] = 6
        self.map8[0, 1, 0] = 7

    def mapSyms(self, reimc: np.ndarray):
        # 8PSK specific, add dimensions
        reimd = reimc.view(np.float32).reshape((-1, 2))
        scaling = np.max(self.eo_metric)  # Assumes eye-opening has been done
        reim_thresh = np.abs(
            np.abs(np.cos(np.pi / 8) * scaling) - np.abs(np.sin(np.pi / 8) * scaling)
        )
        # Compute |X| - |Y|
        xmy = np.abs(reimd[:, 0]) - np.abs(reimd[:, 1])
        # And then | |X| - |Y| | + c, this transforms into QPSK box below XY plane
        # with the new QPSK diamond above XY plane
        z = (
            np.abs(xmy) - reim_thresh
        )  # Do not stack into single array, no difference anyway

        # C1: Check Z > 0; if + check even (diamond), if - check odd (QPSK, box)
        c1z = z > 0

        # C2: Z+ check XY and end, Z- check |X|-|Y| and C3
        cx2 = reimd[:, 0] > 0
        cy2 = reimd[:, 1] > 0
        cxmy2 = xmy > 0

        # C3: + check X, - check Y
        cx3 = np.logical_and(cxmy2, cx2)
        cy3 = np.logical_and(np.logical_not(cxmy2), cy2)

        # Build backwards
        idx1 = cxmy2
        idx2 = np.logical_or(cx3, cy3)

        idx1 = np.logical_or(
            np.logical_and(c1z, idx1), np.logical_and(np.logical_not(c1z), cx2)
        )
        idx2 = np.logical_or(
            np.logical_and(c1z, idx2), np.logical_and(np.logical_not(c1z), cy2)
        )

        idx0 = c1z

        # Now map
        idx = np.vstack(
            (idx0.astype(np.uint8), idx1.astype(np.uint8), idx2.astype(np.uint8))
        )
        # Converts to the default demodulator constellation integers
        syms = self.map8[
            tuple(idx)
        ]  # Needs to be vstack, and need the tuple(); need each value to be a column of indices

        return syms


try:
    import cupy as cp
    import os
    from cupyExtensions import cupyModuleToKernelsLoader, cupyRequireDtype

    # Raw kernel to get many eye openings at once as a batch
    with open(
        os.path.join(
            os.path.dirname(__file__), "custom_kernels", "eyeOpeningKernel.cu"
        ),
        "r",
    ) as fid:
        eyeOpeningBatchKernel = cp.RawKernel(fid.read(), """getEyeOpening_batch""")

    (
        lockPhase_mapSyms_singleBlkKernel_qpsk,
        demod_qpsk_kernel,
        compareIntegerPreambles_kernel,
        cutAndRotate_gray_kernel,
        detectBPSKorQPSK_kernel,
        demod_b_or_q_psk_kernel,
    ), _ = cupyModuleToKernelsLoader(
        "demodulation.cu",
        [
            "lockPhase_mapSyms_singleBlkKernel_qpsk",
            "demod_qpsk",
            "compareIntegerPreambles",
            "cutAndRotatePSKSymbolsFromPossiblePreambles_Gray",
            "detectBPSKorQPSK",
            "demod_b_or_q_psk",
        ],
    )

    # Classes
    class CupyDemodulatorPSK:
        def __init__(self, m: int):
            self.m = m

            # Interrim output
            self.xeo = None  # Selected eye-opening resample points
            self.xeo_i = None  # Index of eye-opening
            self.eo_metric = None  # Metrics of eye-opening
            self.reimc = None  # Phase-locked to constellation (complex array)
            self.svd_metric = None  # SVD metric for phase lock
            self.angleCorrection = None  # Angle correction used in phase lock
            self.syms = None  # Output mapping to each symbol (0 to M-1)
            self.matches = None  # Output from amble rotation search

        @staticmethod
        def demod_b_or_q_psk(
            d_xbatch: cp.ndarray, d_m: cp.ndarray, THREADS_PER_BLOCK: int = 128
        ):
            # Turn into a 2-D if necessary
            if d_xbatch.ndim == 1:
                d_xbatch = d_xbatch.reshape((1, -1))

            # Check dimensions match
            if d_m.ndim != 1:
                raise ValueError("d_m must be 1D.")
            if d_xbatch.shape[0] != d_m.size:
                raise ValueError("d_xbatch must have rows == d_m.size")

            # Check types
            cupyRequireDtype(cp.complex64, d_xbatch)
            cupyRequireDtype(cp.uint8, d_m)

            numSignals, xlength = d_xbatch.shape

            NUM_BLKS = numSignals

            # Allocate shared mem
            smReq = xlength * 8 + THREADS_PER_BLOCK * 8
            cupyCheckExceedsSharedMem(smReq)

            # Allocate output
            d_syms = cp.zeros(d_xbatch.shape, dtype=cp.uint8)

            # Invoke kernel
            demod_b_or_q_psk_kernel(
                (NUM_BLKS,),
                (THREADS_PER_BLOCK,),
                (d_xbatch, xlength, numSignals, d_m, d_syms),
                shared_mem=smReq,
            )

            return d_syms

        @staticmethod
        def _checkEigResults(d_x: cp.ndarray):
            """
            Just meant for debugging purposes for the inner device function.

            Results per row are:
            0-3 : 2x2 row-major matrix from inner product
            4-5 : Bigger eigenvalue then smaller eigenvalue
            6-9 : 2x2 row-major matrix of eigenvectors (6&8) is one column eigenvector, (7&9) is another column eigenvector
            """
            d_eigresults = cp.zeros((d_x.shape[0], 10), dtype=cp.float32)

            THREADS_PER_BLOCK = 128
            NUM_BLKS = d_x.shape[0]
            smReq = THREADS_PER_BLOCK * 16 + d_x.shape[1] * 8

            detectBPSKorQPSK_kernel(
                (NUM_BLKS,),
                (THREADS_PER_BLOCK,),
                (d_x, int(d_x.shape[1]), int(d_x.shape[0]), d_eigresults),
                shared_mem=smReq,
            )

            return d_eigresults

        def getEyeOpening(self, x: cp.ndarray, osr: int, abs_x: cp.ndarray = None):
            if abs_x is None:
                abs_x = cp.abs(
                    x
                )  # Provide option for pre-computed (often used elsewhere anyway)
            x_rs_abs = abs_x.reshape((-1, osr))
            self.eo_metric = cp.mean(x_rs_abs, axis=0)
            i = cp.argmax(self.eo_metric)
            x_rs = x.reshape((-1, osr))
            return x_rs[:, i], i

        def getEyeOpeningBatch(
            self, xbatch: cp.ndarray, osr: int, abs_xbatch: cp.ndarray
        ):

            pass

        @staticmethod
        def prepareIntPreambles(integerPreamblesDict: dict):
            ordering = []
            lengths = []
            amalg = []
            for k, v in integerPreamblesDict.items():
                ordering.append(k)
                lengths.append(v.size)
                amalg.append(v)
            # Convert to gpu arrays
            d_amalg = cp.asarray(np.hstack(amalg), dtype=cp.uint8)

            return ordering, lengths, d_amalg

        @staticmethod
        def compareIntPreambles(
            d_syms: cp.ndarray,
            lengths: np.ndarray,
            d_preamble_concat: cp.ndarray,
            m: int,
            psk_m: cp.ndarray = None,
            searchStart: int = 0,
            searchEnd: int = 128,
            THREADS_PER_BLOCK: int = 128,
        ):
            # Ensure m is reasonable
            if m not in [2, 4, 8]:
                raise ValueError("m must be 2/4/8.")

            # Check that the psk_m mask is reasonable if it is specified
            if psk_m is not None:
                cupyRequireDtype(cp.uint8, psk_m)
                if psk_m.shape != (d_syms.shape[0],):
                    raise ValueError("psk_m shape doesn't match d_syms rows.")
            else:
                psk_m = 0  # Set to null pointer

            # Check types and lengths
            symsLength = d_syms.shape[1]
            numSignals = d_syms.shape[0]

            if np.sum(lengths) != d_preamble_concat.size:
                raise ValueError("Concatenated length is not equal to sum of lengths!")

            if d_preamble_concat.dtype != cp.uint8:
                raise TypeError("Concatenated preamble should be type uint8.")

            if d_syms.dtype != cp.uint8:
                raise TypeError("Symbols matrix should be type uint8.")

            if searchEnd + np.max(lengths) >= symsLength:
                raise ValueError(
                    "Search will extend past the syms length. Shorten the searchEnd."
                )

            # Convert lengths to gpu array
            d_lengths = cp.asarray(lengths, dtype=np.int32)

            # Allocate output
            d_matches = cp.zeros(
                numSignals * len(lengths) * (searchEnd - searchStart) * m,
                dtype=np.uint32,
            )

            # Allocate shared memory
            smReq = (
                d_lengths.nbytes
                + symsLength * 1
                + d_preamble_concat.nbytes
                + (searchEnd - searchStart) * m * 4
            )

            # Invoke kernel
            NUM_BLKS = numSignals
            compareIntegerPreambles_kernel(
                (NUM_BLKS,),
                (THREADS_PER_BLOCK,),
                (
                    d_syms,
                    numSignals,
                    symsLength,
                    searchStart,
                    searchEnd,
                    d_preamble_concat,
                    d_lengths.size,
                    d_lengths,
                    m,
                    d_matches,
                    psk_m,
                ),
                shared_mem=smReq,
            )

            # Reshape matches for viewability
            d_matches = d_matches.reshape(
                (numSignals, d_lengths.size, searchEnd - searchStart, m)
            )

            return d_matches

        @staticmethod
        def cutAndRotateFromPreambles(
            d_argmaxMatches: cp.ndarray,
            d_syms: cp.ndarray,
            d_preambleLengths: cp.ndarray,
            d_sampleStops: cp.ndarray,
            m: int,
            d_psk_m: cp.ndarray = 0,
            outLength: int = None,
            d_out: cp.ndarray = None,
            d_count: cp.ndarray = None,
            THREADS_PER_BLK: int = 128,
            alsoReturnWrittenCounts: bool = False,
        ):
            if isinstance(d_psk_m, cp.ndarray):
                cupyRequireDtype(cp.uint8, d_psk_m)
                if d_psk_m.shape != (d_syms.shape[0],):
                    raise ValueError("d_psk_m must match d_syms rows")

            # Performing checks
            cupyRequireDtype(cp.uint32, d_argmaxMatches)
            cupyRequireDtype(cp.uint32, d_preambleLengths)
            cupyRequireDtype(cp.uint32, d_sampleStops)
            cupyRequireDtype(cp.uint8, d_syms)

            # Get dimensions and check
            numRows, symsLength = d_syms.shape
            if d_argmaxMatches.shape[0] != numRows or d_argmaxMatches.shape[1] != 3:
                raise ValueError("d_argmaxMatches must be %d x 3" % (numRows))
            if d_sampleStops.size != numRows:
                raise ValueError("d_sampleStops must be length %d" % (numRows))

            # Allocate shared mem
            smReq = 3 * 4

            # Allocate output
            if outLength is None:
                outLength = symsLength
                # print("Using outLength = %d" % (outLength))
            if d_out is None:
                d_out = cp.zeros((numRows, outLength), dtype=cp.uint8)
                # print("Allocated output size %d, %d" % (numRows, outLength))

            # Invoke kernel
            NUM_BLKS = numRows
            # print("numRows = %d" % (NUM_BLKS))
            if alsoReturnWrittenCounts:
                if d_count is None:
                    d_count = cp.zeros(numRows, dtype=cp.uint32)

                # print("outLength = %d" % (outLength))
                cutAndRotate_gray_kernel(
                    (NUM_BLKS,),
                    (THREADS_PER_BLK,),
                    (
                        d_argmaxMatches,
                        numRows,
                        d_syms,
                        symsLength,
                        d_preambleLengths,
                        d_sampleStops,
                        np.uint8(m),
                        outLength,
                        d_out,
                        d_count,
                        d_psk_m,
                    ),
                    shared_mem=smReq,
                )

                return d_out, d_count

            else:
                cutAndRotate_gray_kernel(
                    (NUM_BLKS,),
                    (THREADS_PER_BLK,),
                    (
                        d_argmaxMatches,
                        numRows,
                        d_syms,
                        symsLength,
                        d_preambleLengths,
                        d_sampleStops,
                        np.uint8(m),
                        outLength,
                        d_out,
                        0,
                        d_psk_m,
                    ),
                    shared_mem=smReq,
                )

                return d_out

    class CupyDemodulatorQPSK:
        def __init__(
            self,
            batchLength: int,
            numBitsPerBurst: int,
            cluster_threshold: float = 0.1,
            batch_size: int = 4096,
        ):
            self.m = 4
            self.cluster_threshold = cluster_threshold
            self.batch_size = batch_size
            self.batchLength = batchLength
            self.numBitsPerBurst = numBitsPerBurst  # Note that this number is twice the number of symbols used, since QPSK

            # One-time pre-allocation
            self.d_reim_batch = cp.zeros((batch_size, batchLength), dtype=cp.complex64)
            self.d_reimc_batch = cp.zeros((batch_size, batchLength), dtype=cp.complex64)
            self.d_syms_batch = cp.zeros((batch_size, batchLength), dtype=cp.uint32)
            self.d_bestMatches = cp.zeros((batch_size), dtype=cp.int32)
            self.d_bestRotations = cp.zeros((batch_size), dtype=cp.int32)
            self.d_bestMatchIdx = cp.zeros((batch_size), dtype=cp.int32)
            self.d_bits_batch = cp.zeros((batch_size, numBitsPerBurst), dtype=cp.uint8)

            # Counter for batching
            # self.actr = 0 # Batching for eye-opening
            self.bctr = 0  # Batching for demodulation
            # Note that they should be the same at the end

            # # Interrim output
            # self.xeo = None # Selected eye-opening resample points
            # self.xeo_i = None # Index of eye-opening
            # self.eo_metric = None # Metrics of eye-opening
            # self.reimc = None # Phase-locked to constellation (complex array)
            # self.svd_metric = None # SVD metric for phase lock
            # self.angleCorrection = None # Angle correction used in phase lock
            # self.syms = None # Output mapping to each symbol (0 to M-1)
            # self.matches = None # Output from amble rotation search

        @staticmethod
        def demod(d_xbatch: cp.ndarray, THREADS_PER_BLOCK: int = 128):
            if d_xbatch.ndim == 2:
                numSignals, xlength = d_xbatch.shape
            elif d_xbatch.ndim == 1:
                numSignals = 1
                xlength = d_xbatch.size
            else:
                raise ValueError("Input must be 1D or 2D array.")

            NUM_BLKS = numSignals

            # Allocate shared mem
            smReq = xlength * 8 + THREADS_PER_BLOCK * 8
            if smReq > 48000:
                raise MemoryError("Requested shared mem exceeds 48kB: %d bytes" % smReq)

            # Allocate output
            d_syms = cp.zeros(d_xbatch.shape, dtype=cp.uint8)

            # Invoke kernel
            demod_qpsk_kernel(
                (NUM_BLKS,),
                (THREADS_PER_BLOCK,),
                (d_xbatch, xlength, numSignals, d_syms),
                shared_mem=smReq,
            )

            return d_syms

        @staticmethod
        def _getEyeOpeningBatch(
            xbatch: cp.ndarray,
            osr: int,
            abs_xbatch: cp.ndarray,
            d_xeo: cp.ndarray = None,
            count: int = None,
            THREADS_PER_BLOCK: int = 128,
        ):
            """
            This is the static method version of the other method.
            Useful if done standalone.
            """
            # Blocks match the number of signals present in the batch, but you can
            # specify the number of rows if the matrix has unused rows
            NUM_BLOCKS = count if count is not None else xbatch.shape[0]

            # simple shared memory requirements
            smReq = THREADS_PER_BLOCK * osr * 4

            # Allocate output if not given
            if d_xeo is None:
                d_xeo = cp.zeros(
                    (NUM_BLOCKS, xbatch.shape[1] // osr), dtype=cp.complex64
                )
            else:
                # Check the data type and length
                if d_xeo.dtype != cp.complex64:
                    raise TypeError("d_xeo must be complex64.")
                if d_xeo.shape[1] < xbatch.shape[1] // osr:
                    raise ValueError(
                        "d_xeo must have at least %d columns."
                        % (xbatch.shape[1] // osr)
                    )

            # Invoke kernel
            eyeOpeningBatchKernel(
                (NUM_BLOCKS,),
                (THREADS_PER_BLOCK,),
                (abs_xbatch, abs_xbatch.shape[1], osr, xbatch, d_xeo),
                shared_mem=smReq,
            )

            return d_xeo

        def getEyeOpeningBatch(
            self,
            xbatch: cp.ndarray,
            osr: int,
            abs_xbatch: cp.ndarray,
            count: int = None,
        ):
            THREADS_PER_BLOCK = 128
            NUM_BLOCKS = count if count is not None else xbatch.shape[0]
            # Update the counter
            self.bctr = NUM_BLOCKS

            # simple shared memory requirements
            smReq = THREADS_PER_BLOCK * osr * 4

            # Invoke kernel
            eyeOpeningBatchKernel(
                (NUM_BLOCKS,),
                (THREADS_PER_BLOCK,),
                (abs_xbatch, abs_xbatch.shape[1], osr, xbatch, self.d_reim_batch),
                shared_mem=smReq,
            )

        def getEyeOpening(self, x: cp.ndarray, osr: int, abs_x: cp.ndarray):
            ## This is just a straight np to cp conversion.. Verified!

            x_rs_abs = abs_x.reshape((-1, osr))
            self.eo_metric = cp.sum(x_rs_abs, axis=0)
            # i = cp.argmax(self.eo_metric)
            # i = np.argmax(self.eo_metric.get()) # This is definitely slower
            # x_rs = x.reshape((-1, osr))
            # return x_rs[:,i], i

        def gather(self, reim: cp.ndarray):
            self.d_reim_batch[self.bctr, :] = reim
            self.bctr = self.bctr + 1

        def resetBatch(self):
            self.bctr = 0

        @staticmethod
        def _demodBatch(
            d_xeo: cp.ndarray,
            amble: cp.ndarray,
            numBitsPerBurst: int,
            searchStart: int = 0,
            searchlength: int = 128,
            THREADS_PER_BLOCK: int = 128,
        ):
            """
            Static, standalone method of the other method of the same name.
            """
            NUM_BLOCKS = d_xeo.shape[0]

            # Check shared memory requirements
            workspaceSize = np.max(
                [THREADS_PER_BLOCK * 4, (searchlength * 2 + amble.size) * 4]
            )
            smReq = (
                d_xeo.shape[1] * 8 + workspaceSize
            )  # THREADS_PER_BLOCK * 4 + reim.nbytes
            if smReq > 48000:
                raise MemoryError("Shared memory requested exceeded 48kB.")

            # Allocations
            batch_size = d_xeo.shape[0]
            d_reimc_batch = cp.zeros_like(d_xeo)
            d_syms_batch = cp.zeros(d_xeo.shape, dtype=cp.uint32)
            d_bestMatches = cp.zeros((batch_size), dtype=cp.int32)
            d_bestRotations = cp.zeros((batch_size), dtype=cp.int32)
            d_bestMatchIdx = cp.zeros((batch_size), dtype=cp.int32)
            d_bits_batch = cp.zeros((batch_size, numBitsPerBurst), dtype=cp.uint8)

            lockPhase_mapSyms_singleBlkKernel_qpsk(
                (NUM_BLOCKS,),
                (THREADS_PER_BLOCK,),
                (
                    d_xeo,
                    d_xeo.shape[1],
                    amble,
                    amble.size,
                    searchStart,
                    searchlength,
                    d_reimc_batch,
                    d_syms_batch,
                    d_bestMatches,
                    d_bestRotations,
                    d_bestMatchIdx,
                    d_bits_batch,
                    numBitsPerBurst,
                ),
                shared_mem=smReq,
            )

            return (
                d_reimc_batch,
                d_syms_batch,
                d_bestMatches,
                d_bestRotations,
                d_bestMatchIdx,
                d_bits_batch,
            )

        def demodBatch(
            self, amble: cp.ndarray, searchStart: int = 0, searchlength: int = 128
        ):
            THREADS_PER_BLOCK = 128
            NUM_BLOCKS = self.bctr

            # Check shared memory requirements
            workspaceSize = np.max(
                [THREADS_PER_BLOCK * 4, (searchlength * 2 + amble.size) * 4]
            )
            smReq = (
                self.batchLength * 8 + workspaceSize
            )  # THREADS_PER_BLOCK * 4 + reim.nbytes
            if smReq > 48000:
                raise MemoryError("Shared memory requested exceeded 48kB.")

            lockPhase_mapSyms_singleBlkKernel_qpsk(
                (NUM_BLOCKS,),
                (THREADS_PER_BLOCK,),
                (
                    self.d_reim_batch,
                    self.batchLength,
                    amble,
                    amble.size,
                    0,
                    128,
                    self.d_reimc_batch,
                    self.d_syms_batch,
                    self.d_bestMatches,
                    self.d_bestRotations,
                    self.d_bestMatchIdx,
                    self.d_bits_batch,
                    self.numBitsPerBurst,
                ),
                shared_mem=smReq,
            )

            return (
                self.d_reimc_batch,
                self.d_syms_batch,
                self.d_bestMatches,
                self.d_bestRotations,
                self.d_bestMatchIdx,
                self.d_bits_batch,
            )

        # def demod(self, reim: cp.ndarray, amble: cp.ndarray, searchStart: int=0, searchlength: int=128):
        #     # Allocate output
        #     d_reimc = cp.zeros(reim.size, dtype=cp.complex64)
        #     d_syms = cp.zeros(reim.size, dtype=cp.int32)
        #     d_matches = cp.zeros(searchlength, dtype=cp.int32)
        #     d_rotation = cp.zeros(searchlength, dtype=cp.int32)

        #     THREADS_PER_BLOCK = 128
        #     NUM_BLOCKS = 1

        #     # Check shared memory requirements
        #     workspaceSize = np.max([THREADS_PER_BLOCK * 4, (searchlength*2 + amble.size)*4])
        #     smReq = reim.nbytes + workspaceSize # THREADS_PER_BLOCK * 4 + reim.nbytes
        #     if smReq > 48000:
        #         raise MemoryError("Shared memory requested exceeded 48kB.")

        #     lockPhase_mapSyms_singleBlkKernel_qpsk((NUM_BLOCKS,),(THREADS_PER_BLOCK,),
        #                        (reim, reim.size, amble, amble.size, 0, 128, d_reimc, d_syms, d_matches, d_rotation),
        #                        shared_mem=smReq)

        #     return d_reimc, d_syms, d_matches, d_rotation

        def symsToBits(self, syms: np.ndarray = None):
            pass

        def unpackToBinaryBytes(self, packed: np.ndarray):
            pass

        def packBinaryBytesToBits(self, unpacked: np.ndarray):
            pass

except Exception as e:
    print("%s\nSkipping imports of cupy demodulators." % str(e))


# %%
# @jit(nopython=True)
def demodulateCP2FSK(syms, h, up):
    m = np.array([[-1], [+1]])  # these map to [0, 1] bits

    # create the two tones
    tones = np.exp(1j * np.pi * h * np.arange(up) / up * m)

    # loop over each upsampled section of a symbol
    numSyms = int(np.floor(len(syms) / up))
    bitCost = np.zeros((2, numSyms))
    demodBits = np.zeros(numSyms, dtype=np.uint8)

    for i in range(numSyms):
        symbol = syms[i * up : (i + 1) * up]

        for k in range(2):
            bitCost[k, i] = np.abs(np.vdot(symbol, tones[k]))

        demodBits[i] = np.argmax(bitCost[:, i])

    return demodBits, bitCost, tones


##
class BurstyDemodulator:
    """
    This class and all derived versions attempt to perform an aligned, synchronous
    demodulation of all bursts at the same time, amalgamating the cost functions into one.
    This prevents the misalignment (usually by one symbol) of the bursts, which when used
    in remodulation of the observed signal, can cause grave reconstruction errors in the
    differential modulation modes (DQPSK, CPFSK etc.).
    """

    def __init__(self, burstLen: int, guardLen: int, up: int = 1):
        """
        Note that burstLen and guardLen are the number of bits.
        Do not include the OSR (this will be accounted for by the 'up' parameter.
        """
        self.burstLen = burstLen
        self.guardLen = guardLen
        self.period = self.burstLen + self.guardLen
        self.up = up

    def demod(self, x: np.ndarray, numBursts: int, searchIdx: np.ndarray = None):
        raise NotImplementedError("Only invoke with derived classes.")


##
class BurstyDemodulatorCP2FSK(BurstyDemodulator):
    def __init__(self, burstLen: int, guardLen: int, up: int = 1, h: float = 0.5):
        super().__init__(burstLen, guardLen, up)  # Refer to parent class
        # Extra params
        self.h = h

        # Configurations
        self.burstIdxs = None

        # Outputs
        self.d_costs = None
        self.searchIdx = None

    def demod(self, x: np.ndarray, numBursts: int = None, searchIdx: np.ndarray = None):
        # t1 = time.perf_counter()
        if self.burstIdxs is None:  # Priority is to use the pre-set burstIdxs
            if (
                numBursts is None
            ):  # Otherwise we generate it here using a simple np.arange
                raise ValueError(
                    "Please call setBurstIdxs() before demodulating or set the numBursts argument."
                )
            else:
                self.setBurstIdxs(np.arange(numBursts))

        # Construct the tones
        gtone = np.exp(1j * np.pi * self.h * np.arange(self.up) / self.up)
        tones = np.vstack((gtone.conj(), gtone))
        # Perform a one-pass correlation over the entire array
        xc = np.vstack([np.correlate(x, tone) for tone in tones])
        xc_abs = np.abs(xc)
        # Also perform the one-pass max over it
        xc_abs_argmax = np.argmax(xc_abs, axis=0)
        xc_abs_max = np.max(
            xc_abs, axis=0
        )  # Note: using the argmax to regenerate using list comprehension is extremely slow..

        # Construct the starting indices for each burst
        burstStarts = self.burstIdxs * self.period * self.up

        # Construct the symbol spacing for each individual burst
        symbolSpacing = np.arange(0, self.burstLen * self.up, self.up)
        # Construct zero-origin indices for every symbol, in every burst
        genIdx = np.array([start + symbolSpacing for start in burstStarts]).flatten()

        # Construct a search index if not specified
        if searchIdx is None:
            # extent = genIdx[-1] + self.up
            searchIdx = np.arange(xc_abs_max.size - genIdx[-1])
            # print("Auto-generated search indices from %d to %d" % (searchIdx[0], searchIdx[-1]))

        # Loop over the search range
        self.d_costs = np.zeros(searchIdx.size)
        for i, s in enumerate(searchIdx):
            # Construct the shifted indices for every symbol based on current search index
            idx = s + genIdx
            # Sum the costs for these indices
            self.d_costs[i] = np.sum(xc_abs_max[idx])

        # Now find the best cost and extract the corresponding bits
        mi = searchIdx[np.argmax(self.d_costs)]
        dbits = xc_abs_argmax[mi + genIdx].reshape((-1, self.burstLen))

        # t2 = time.perf_counter()
        # print("Took %f s." % (t2-t1))

        # Save searchIdx for plotting
        self.searchIdx = searchIdx

        return dbits, mi

    def setBurstIdxs(self, burstIdxs: np.ndarray = None):
        """
        Parameters
        ----------
        burstIdxs : np.ndarray, optional
            Integer array of burst indices that should be demodulated.
            This can be used to ignore certain bursts, or if there are missing bursts
            within the window passed to the demod() call.
            The default is None, which will then fit as many bursts as possible during
            the demod() method call.

        Returns
        -------
        None.

        """
        self.burstIdxs = burstIdxs

    def plotCosts(self):
        fig, ax = plt.subplots(1, 1, num="CP2FSK Demodulator Cost")
        ax.plot(self.searchIdx, self.d_costs)
        return fig, ax


# %%
def convertIntToBase4Combination(l, i):
    base_4_repr = np.array(list(np.base_repr(i, base=4)), dtype=np.uint8)
    base_4_repr = np.pad(
        base_4_repr, (l - len(base_4_repr), 0)
    )  # pad it to the numSyms

    return base_4_repr


def ML_demod_QPSK(y, h, up, numSyms):
    """
    Brute force search over all combinations, don't use this if you have a long string of symbols.
    """

    totalCombi = 4**numSyms
    cost = np.zeros(totalCombi)

    for i in range(totalCombi):
        base_4_repr = convertIntToBase4Combination(numSyms, i)

        qpsk_syms = np.exp(1j * base_4_repr * np.pi / 2)

        qpsk_syms_up = np.zeros(numSyms * up, dtype=np.complex128)
        qpsk_syms_up[::up] = qpsk_syms

        # convolve this test symbol set with the channel
        test = np.convolve(h, qpsk_syms_up)

        # # find the normalised dot product
        # test = test[up : up + len(y)]
        # test_cost = np.vdot(y, test) / np.linalg.norm(y) / np.linalg.norm(test)
        # cost[i] = np.abs(test_cost)**2.0

        # find the smallest norm difference
        test = test[up : up + len(y)]
        test_cost = np.linalg.norm(test - y)
        cost[i] = -test_cost  # minus to maintain the maximization criterion

        # print(base_4_repr)
        # print(qpsk_syms_up)
        # print(test)

        # if np.all(base_4_repr == np.array([0,1,2,1,2,3])):
        #     print(base_4_repr)
        #     print(test)
        #     print(qpsk_syms_up)
        #     break

    ii = np.argmax(cost)
    mm = convertIntToBase4Combination(numSyms, ii)

    return mm, ii, cost


# def ML_demod_CPM_laurent(y, h, up, numSyms, outputCost=False):
#     '''
#     Brute force search over all combinations, don't use this if you have a long string of symbols.
#     '''

#     totalCombi = 2**numSyms
#     cost = np.zeros(totalCombi)
#     rowsToKeep = int(np.ceil(numSyms)/8)

#     for i in range(totalCombi):
#         i_arr = np.array([i], np.uint32) # convert to array
#         i_arr = i_arr.view(np.uint8).reshape((-1,1)) # convert to byte level
#         i_bits = np.unpackbits(i_arr, axis=1, bitorder='little')

#         i_bits = i_bits[:rowsToKeep+1].flatten()
#         i_bits = i_bits[0:numSyms]

#         print(i_bits)

#         for b in range(len(i_bits)):

# %% Workspace


# Leaving this reference code here for 2x2 eigvalues, to be converted into cuda kernels
# Useful shortcut: https://people.math.harvard.edu/~knill/teaching/math21b2004/exhibits/2dmatrices/index.html
def eig2x2(x, normaliseDet=True):
    # For better numerical accuracy, ensure determinant is 1.0 by doing this
    if normaliseDet:
        nf = np.linalg.det(x) ** 0.5
        x = x / nf

    a = 1.0
    b = -x[0, 0] - x[1, 1]
    print(b)
    c = x[0, 0] * x[1, 1] - x[0, 1] * x[1, 0]
    print(c)
    f = np.sqrt(b * b - 4 * a * c) / (2 * a)
    xp = -b / (2 * a) + f
    xm = -b / (2 * a) - f

    e1 = np.array([[x[0, 1]], [xp - x[0, 0]]])
    e2 = np.array([[x[0, 1]], [xm - x[0, 0]]])
    # second way
    e1 = np.array([[xp - x[1, 1]], [x[0, 1]]])
    e2 = np.array([[xm - x[1, 1]], [x[0, 1]]])

    return xp, xm, e1, e2


# %% Unit testing

if __name__ == "__main__":
    from signalCreationRoutines import *
    from plotRoutines import *
    import matplotlib.pyplot as plt
    from timingRoutines import Timer

    closeAllFigs()

    OSR = 8
    numBits = 10000
    m = 4
    syms, bits = randPSKsyms(numBits, m)
    syms_rs = sps.resample_poly(syms, OSR, 1)
    _, rx = addSigToNoise(
        numBits * OSR, 0, syms_rs, chnBW=OSR, snr_inband_linear=1000
    )  # Inf SNR simulates perfect filtering
    randomPhase = np.random.rand() * (2 * np.pi / m)  # Induce random phase
    rx = rx * np.exp(1j * randomPhase)
    ofig, oax = plt.subplots(2, 1, num="Original")
    plotConstellation(syms_rs, ax=oax[0])
    plotConstellation(rx, ax=oax[0])
    plotSpectra([rx], [1], ax=oax[1])

    # Divide into resampled portions
    rxrs = rx.reshape((-1, OSR))
    fig, ax = plt.subplots(OSR // 2, OSR // 2)
    for i in range(OSR):
        plotConstellation(rxrs[:, i], ax=ax[i // 2, i % 2])

    # Projection to log2(m)+1 dimensions
    demodulator = SimpleDemodulatorPSK(m)

    # Most generic demodulator shouldn't use angle to avoid if/else costs
    # Use the normalised dot product from constellation, then find max across rows
    timer = Timer()
    timer.start()
    genericSyms = demodulator.demod(rx.astype(np.complex64), OSR)
    timer.end("Default demodulator")
    rotationalInvariance = (genericSyms - bits) % m
    rax = plotConstellation(demodulator.reimc)
    assert np.all(rotationalInvariance == rotationalInvariance[0])

    unpackedSyms = demodulator.unpackToBinaryBytes(genericSyms)
    demodbits = demodulator.symsToBits()

    # Test preamble detection
    preamble = bits[400 : 400 + 32]
    timer.start()
    rotatedSyms, sample, rotation = demodulator.ambleRotate(
        preamble, np.arange(400, 400 + 256, dtype=np.int32)
    )
    timer.end("Preamble search")
    plt.figure("Preamble matching")
    plt.plot(demodulator.matches)

    # Unit test on pure QPSK
    if m == 4:
        pure4 = demodulator.pskdicts[4]

        rxScaled = rx.astype(np.complex64) * 3
        rxScaledAbs = np.abs(
            rxScaled
        )  # Providing the amplitude pre-computed reduces computation significantly

        timer.reset()
        demod4 = SimpleDemodulatorQPSK()  # Custom symbol mapper is present, about 3-4x
        timer.start()
        # demod4.mapSyms(pure4)
        # demod4.mapSyms(demodulator.reimc)
        genericSyms4 = demod4.demod(rxScaled, OSR, abs_x=rxScaledAbs)
        timer.end("QPSK demodulator")
        rotationalInvariance4 = (genericSyms4 - bits) % m
        plotConstellation(demod4.reimc, ax=rax)
        assert np.all(rotationalInvariance4 == rotationalInvariance4[0])

    # Unit test on pure 8PSK
    if m == 8:
        pure8 = demodulator.pskdicts[8]

        timer.reset()
        demod8 = SimpleDemodulator8PSK()  # Custom symbol mapper is present, about 2-4x
        timer.start()
        # demod8.mapSyms(pure8)
        # demod8.mapSyms(demodulator.reimc)
        genericSyms8 = demod8.demod(rx.astype(np.complex64) * 3, OSR)
        timer.end("8PSK demodulator")
        rotationalInvariance8 = (genericSyms8 - bits) % m
        assert np.all(rotationalInvariance8 == rotationalInvariance8[0])

    # %%
    # baud = 10000
    # up = 10
    # fs = baud * up

    # numBursts = 99
    # burstBits = 90
    # period = 100
    # guardBits = period - burstBits

    # bits = randBits(burstBits * numBursts, 2).reshape((numBursts,burstBits))
    # sigs = []
    # for i in range(numBursts):
    #     sig, _, _, _ = makePulsedCPFSKsyms(bits[i,:], baud, up=up)
    #     sigs.append(sig)

    # sigStart = int(0.25*sigs[0].size)
    # snr = 10
    # _, rx = addManySigToNoise(int((numBursts+1)*period*up),
    #                           np.arange(0,numBursts*period*up,period*up)+sigStart, sigs, bw_signal = baud, chnBW = fs,
    #                    snr_inband_linearList = snr+np.zeros(numBursts))

    # fig, ax = plt.subplots(2,1)
    # ax[0].plot(np.abs(rx))
    # plotSpectra([rx],[fs],ax=ax[1])

    # # Filter
    # taps = sps.firwin(201, baud/fs*1.5)
    # rxfilt = sps.lfilter(taps,1,rx)
    # ax[0].plot(np.abs(rxfilt))
    # plotSpectra([rxfilt],[fs],ax=ax[1])

    # # Attempt bursty demod
    # searchRange = np.arange(int(0.5*sigs[0].size))

    # bd = BurstyDemodulatorCP2FSK(burstBits, (period-burstBits), up)
    # dbits, dalign = bd.demod(rxfilt, numBursts, searchRange)
    # print("Demodulation index at %d" % dalign)
    # plt.figure("Bursty demod cost")
    # plt.plot(searchRange, bd.dcosts)

    # # # This is the exact actual value
    # # dalign = 327
    # # dbits = bd.demodAtIdx(rxfilt, dalign, numBursts, numBursts * period * up - guardBits * up) # Hard coded correct alignment

    # if np.all(dbits==bits):
    #     print("Full demodulation of %d bursts * %d symbols is correct." % (numBursts, burstBits))

    # else:
    #     print("Full demodulation failed for these bursts:")
    #     failedBursts = np.argwhere(np.any(bits != dbits, axis=1)).flatten()
    #     for i in range(failedBursts.size):
    #         fb = failedBursts[i]
    #         rm, _, _, _ = makePulsedCPFSKsyms(dbits[fb,:], baud, up=up)
    #         rm = rm[:burstBits * up] # cut it
    #         balign = dalign + fb * period * up
    #         metric = np.abs(np.vdot(rm, rxfilt[balign:balign+rm.size]))**2 / np.linalg.norm(rm)**2 / np.linalg.norm(rxfilt[balign:balign+rm.size])**2
    #         print("Burst %d with QF2 %f, bits %d/%d" % (fb, metric, np.argwhere(bits[fb,:] == dbits[fb,:]).size, burstBits))

    #     print("Full demodulation successful for these bursts:")
    #     successBursts = np.argwhere(np.all(bits == dbits, axis=1)).flatten()
    #     for i in range(successBursts.size):
    #         sb = successBursts[i]
    #         rm, _, _, _ = makePulsedCPFSKsyms(dbits[sb,:], baud, up=up)
    #         rm = rm[:burstBits * up] # cut it
    #         balign = dalign + sb * period * up
    #         metric = np.abs(np.vdot(rm, rxfilt[balign:balign+rm.size]))**2 / np.linalg.norm(rm)**2 / np.linalg.norm(rxfilt[balign:balign+rm.size])**2
    #         print("Burst %d with QF2 %f, bits %d/%d" % (sb, metric, np.argwhere(bits[sb,:] == dbits[sb,:]).size, burstBits))
