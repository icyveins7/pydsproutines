"""
Some notes on the 2 classes.

The outputs are not necessarily the same as GPU floating point multiplication
is not identical to the CPU's. As such, there may be sporadic extra values which pass
the threshold, and this is more pronounced the longer the original input array is.

"""

import numpy as np
import cupy as cp
import warnings

from cupyHelpers import (
    cupyModuleToKernelsLoader, cupyRequireDtype,
    cupyGetEnoughBlocks, cupyCheckExceedsSharedMem
)
from filterRoutines import cupyMovingAverage, cupyComplexMovingSum

# ============== Basic Pythonic Class


class MatrixProfile:
    def __init__(
        self,
        windowLength: int,
        outputChains: bool = False,
        minThreshold: float = None,
        minChainLength: int = 0
    ):
        self._windowLength = windowLength
        self._normsSq = None
        self._outputChains = outputChains
        if outputChains and minThreshold is None:
            raise ValueError(
                "minThreshold cannot be None if outputChains is True")
        self._minThreshold = minThreshold
        self._minChainLength = minChainLength

    def compute(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the matrix profile.

        Parameters
        ----------
        x : np.ndarray
            The time series.

        Returns
        -------
        mp : list of np.ndarray
            The matrix profile, returned as a list of diagonals.
        """
        self._normsSq = None  # Reset values
        self._normsSq = self._computeNormsSq(x)

        if self._outputChains:
            chains = self._compute_chains(x)
            return chains

        else:
            mp = self._compute_raw(x)
            return mp

    def _compute_raw(self, x: np.ndarray):
        # Loop over all diagonals
        mp = list()
        for i in range(1, x.size - self._windowLength + 1):
            # Compute the diagonal
            diag = self._computeDiagonal(x, i)
            mp.append(diag)

        return mp

    def _compute_chains(self, x: np.ndarray):
        chains = list()
        # Loop over all diagonals
        for i in range(1, x.size - self._windowLength + 1):
            # Compute the diagonal
            diag = self._computeDiagonal(x, i)
            # Extract values which pass threshold
            idx = np.argwhere(diag > self._minThreshold).reshape(-1)
            chainStarts, chainEnds, chainLengths = self._chainify(
                idx, self._minChainLength)

            for j, (cStart, cLen) in enumerate(
                zip(chainStarts, chainLengths)
            ):
                chains.append(
                    (
                        i,
                        idx[cStart],
                        idx[cStart] + cLen
                    )
                )

        return chains

    def _chainify(self, idx_arr: np.ndarray, minChainLength: int = 0):
        """
        Extracts contiguous chains from an indices array.

        Example:
            [1,2,3,7,10,11]
            Outputs should be:
                [0:3], [3:4], [4:6] with lengths
                    3,     1,     2 respectively.

        Parameters
        ----------
        idx_arr : np.ndarray
            Input array of indices. Usually computed from a minimum
            threshold over the diagonal's output.

        minChainLength : int, default=0
            The minimum chain length for a chain to be stored.

        Returns
        -------
        chainStarts : np.ndarray
            Starting index of each chain, with reference to idx_arr.

        chainEnds : np.ndarray
            Ending index of each chain, with reference to idx_arr.

        chainLengths : np.ndarray
            Length of each chain.
        """
        d = np.diff(idx_arr)
        # Find the places where it's not incrementing i.e. jumps
        ii = np.argwhere(d > 1).reshape(-1)
        ii += 1  # Add 1 back since we took the diff
        starts = np.hstack((0, ii))
        ends = np.hstack((ii, idx_arr.size))
        chainLengths = ends - starts
        # Remove any chains that are too short
        selectedIdx = np.argwhere(chainLengths > minChainLength).reshape(-1)
        chainStarts = starts[selectedIdx]
        chainEnds = ends[selectedIdx]
        chainLengths = chainLengths[selectedIdx]

        return chainStarts, chainEnds, chainLengths

    def _computeNormsSq(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the moving window energy of the input,
        defined as norm(x[n:n+windowLength])**2.

        Parameters
        ----------
        x : np.ndarray, length N
            Input array.

        Returns
        -------
        normsSq : np.ndarray, length N-windowLength+1
            The moving window energy.
        """
        power = np.abs(x)**2
        normsSq = np.convolve(
            power, np.ones(self._windowLength), mode='valid')
        return normsSq

    def _computeDiagonal(
        self,
        x: np.ndarray,
        diagIdx: int
    ) -> np.ndarray:
        """
        Compute the diagonal of the matrix profile.

        Parameters
        ----------
        x : np.ndarray, length N
            The time series.
        diagIdx : int
            The index of the diagonal to compute, from 1 to N.

        Returns
        -------
        kdiag : np.ndarray
            The k-diagonal of the matrix profile, corresponding to
            diagIdx.
        """
        if diagIdx <= 0:
            raise ValueError('diagIdx must be greater than 0.')

        slice1 = x[0:-diagIdx]
        slice2 = x[diagIdx:]
        if slice1.size <= 0 or slice2.size <= 0:
            raise RuntimeError("Nothing in the diagonal.")

        energy1 = self._normsSq[0:-diagIdx]
        energy2 = self._normsSq[diagIdx:]
        if energy1.size <= 0 or energy2.size <= 0:
            raise RuntimeError("Indexing out of valid normSq values.")

        pdt = slice1 * slice2.conj()
        kdiag = np.convolve(pdt, np.ones(self._windowLength), mode='valid')
        kdiag = np.abs(kdiag)**2
        # Normalise appropriately
        kdiag = kdiag / energy1 / energy2

        return kdiag


# ===================== CuPy subclass

(
    _matrixProfileChainsKernel,
    _matrixProfileRawKernel
), _ = cupyModuleToKernelsLoader(
    "matrixProfile.cu",
    ["matrix_profile_chains", "matrix_profile_raw"])


class CupyMatrixProfile(MatrixProfile):
    MAX_CHAINS = 10000000

    def setMaxChains(self, maxChains: int):
        """
        Sets the maximum number of chains that can be stored
        during a single compute() invocation.

        This is required in this CUDA kernel-flavoured implementation,
        since the memory is pre-allocated and then atomically filled.

        Hence if there is not enough memory to store all chains, an
        unrecoverable memory error is likely to occur.

        Set it to something big!
        """
        self.MAX_CHAINS = maxChains

    def _compute_raw(self, x: cp.ndarray):
        cupyRequireDtype(cp.complex64, x)
        cupyRequireDtype(cp.float32, self._normsSq)

        # Allocate enough space for whole output
        totalRawSize = (x.size - self._windowLength + 1) * \
            (x.size - self._windowLength) // 2
        # print("Output size: %d" % (totalRawSize))  # DEBUG
        d_out = cp.zeros(totalRawSize, dtype=cp.float32)

        # Determine shared memory requirements
        NUM_PER_THREAD = 33
        THREADS_PER_BLK = 32
        smReq = (NUM_PER_THREAD * THREADS_PER_BLK *
                 2 + self._windowLength-1) * 8
        # smReq = (NUM_PER_THREAD * THREADS_PER_BLK *
        #          3 + self._windowLength-1) * 8  # DEBUG OVERALLOCATE
        # print("Shared mem: %d bytes" % (smReq))  # DEBUG
        cupyCheckExceedsSharedMem(smReq)

        # Determine blocks for full raw calculation
        blocksPerDiagonal = cupyGetEnoughBlocks(
            x.size - self._windowLength,  # length of the k=1 diagonal output
            NUM_PER_THREAD * THREADS_PER_BLK
        )
        # DEBUG
        print("Allocating %d blocksPerDiagonal" % (blocksPerDiagonal))
        # Many extraneous blocks for later diagonals, which will do nothing
        NUM_BLKS = blocksPerDiagonal * (x.size - self._windowLength)
        print("GRID (%d * %d BLKS)" % (NUM_BLKS, blocksPerDiagonal))

        # Invoke kernel
        print("Invoking raw matrix profile kernel")
        _matrixProfileRawKernel(
            (NUM_BLKS,), (THREADS_PER_BLK,),
            (
                x,
                self._normsSq,
                x.size,
                blocksPerDiagonal,
                self._windowLength,
                NUM_PER_THREAD,
                d_out
            ),
            shared_mem=smReq
        )

        return d_out

    def _compute_chains(self,
                        x: cp.ndarray,
                        diagIdx: cp.ndarray = None) -> cp.ndarray:
        """
        Overloaded submethod for computing chains using custom kernel.
        """

        cupyRequireDtype(cp.complex64, x)
        cupyRequireDtype(cp.float32, self._normsSq)
        # Do all diagonals by default
        if diagIdx is None:
            diagIdx = cp.arange(1, x.size - self._windowLength + 1)
        else:
            cupyRequireDtype(cp.int32, diagIdx)

        # Set up to some maximum number of chains for the output
        d_chains = cp.zeros(3*self.MAX_CHAINS, dtype=cp.int32)
        print("Interrim d_chains size = %d" % (d_chains.size))  # DEBUG

        # Determine shared memory requirements
        NUM_PER_THREAD = 33
        THREADS_PER_BLK = 32
        smReq = (NUM_PER_THREAD * THREADS_PER_BLK *
                 2 + self._windowLength-1) * 8
        cupyCheckExceedsSharedMem(smReq)

        # We determine number of blocks by using the longest diagonal
        # i.e. the first one
        blocksPerDiagonal = cupyGetEnoughBlocks(
            x.size - self._windowLength,  # length of the k=1 diagonal output
            NUM_PER_THREAD * THREADS_PER_BLK
        )
        print("blocksPerDiagonal = %d" % (blocksPerDiagonal))
        # Many extraneous blocks for later diagonals, which will do nothing
        NUM_BLKS = blocksPerDiagonal * (x.size - self._windowLength)
        print("NUM_BLKS = %d" % (NUM_BLKS))

        # Allocate counters for atomics
        numChains = cp.zeros(1, cp.int32)

        # Invoke kernel
        print("Invoking chains matrix profile kernel with minThreshold %f" %
              (self._minThreshold))
        _matrixProfileChainsKernel(
            (NUM_BLKS,), (THREADS_PER_BLK,),
            (
                x,
                self._normsSq,
                x.size,
                diagIdx,
                blocksPerDiagonal,
                self._windowLength,
                NUM_PER_THREAD,
                # must decorate type for float32 in order to work
                cp.float32(self._minThreshold),
                cp.int32(self.MAX_CHAINS),
                numChains,
                d_chains
            ),
            shared_mem=smReq
        )
        print(numChains)
        if numChains[0] >= self.MAX_CHAINS:
            warnings.warn(
                "Exceeded maximum number of chains (%d) requested." % self.MAX_CHAINS)

        # Extract the subchains
        subchains = d_chains[:3*numChains[0]].reshape((-1, 3))

        # Connect them then return
        return self._connectSubchains(subchains)

    def _connectSubchains(self, d_chains: cp.ndarray):
        """
        Used to connect the subchains on the host(CPU) side.
        Usually called using the output of _compute_chains.
        """
        # We perform the transpose in the GPU before pulling
        h_chains = cp.ascontiguousarray(d_chains.T).get()
        # It's now 3 rows X N

        # We output as a flat list
        out = list()
        keys = np.unique(h_chains[0])  # Find all unique keys from 1st row
        for k in keys:
            # Extract those in the diagonal
            starts = h_chains[1, h_chains[0] == k]
            ends = h_chains[2, h_chains[0] == k]

            # Then we sort
            sIdx = np.argsort(starts)
            starts = starts[sIdx]
            ends = ends[sIdx]

            # Iterate forwards and connect
            for i, (start, end) in enumerate(zip(starts, ends)):
                # First iteration
                if i == 0:
                    out.append([k, start, end])
                # Check if start matches previous end
                elif start == out[-1][2]:
                    # Amend in place
                    out[-1][2] = end
                # Otherwise create new one
                else:
                    out.append([k, start, end])

        return out

    def _computeNormsSq(self, x: cp.ndarray) -> cp.ndarray:
        # We already have a movingAverage/movingSum kernel, so let's use it
        power = cp.abs(x)**2
        normsSq = cupyMovingAverage(power, self._windowLength, sumInstead=True)
        normsSq = normsSq[self._windowLength-1:]  # Remove the padded zeroes

        return normsSq

    def _computeDiagonal(self, x: cp.ndarray, diagIdx: int) -> cp.ndarray:
        if diagIdx <= 0:
            raise ValueError('diagIdx must be greater than 0.')

        slice1 = x[0:-diagIdx]
        slice2 = x[diagIdx:]
        if slice1.size <= 0 or slice2.size <= 0:
            raise RuntimeError("Nothing in the diagonal.")

        energy1 = self._normsSq[0:-diagIdx]
        energy2 = self._normsSq[diagIdx:]
        if energy1.size <= 0 or energy2.size <= 0:
            raise RuntimeError("Indexing out of valid normSq values.")

        pdt = slice1 * slice2.conj()
        # Here we need to apply the movingAvg kernel but as a complex..
        kdiag = cupyComplexMovingSum(
            pdt, self._windowLength
        )
        # Normalise appropriately
        kdiag = kdiag / energy1 / energy2

        return kdiag


# ==================== Unit testing
if __name__ == "__main__":
    # The following are used for unittests and signal gen
    import unittest
    from xcorrRoutines import calcQF2
    from signalCreationRoutines import randPSKsyms, addManySigToNoise
    from verifyRoutines import compareValues

    # These are used for example code and plotting only
    import pyqtgraph as pg
    from plotRoutines import pgPlotAmpTime, closeAllFigs
    closeAllFigs()

    windowLength = 10
    numCopies = 3
    syms, bits = randPSKsyms(windowLength, 4, dtype=np.complex64)
    noise, x = addManySigToNoise(
        20000,
        np.arange(numCopies) * windowLength * 3,
        [syms for i in range(numCopies)],
        1, 1, [40.0 for i in range(numCopies)]
    )

    mpo = MatrixProfile(windowLength)
    mpo._normsSq = mpo._computeNormsSq(x)
    k = 30
    diag0 = mpo._computeDiagonal(x, k)
    print(diag0)
    print(diag0.shape)

    # awin, aax = pgPlotAmpTime(x, title='ampl-time')

    # win = pg.plot(np.arange(diag0.size) + k, diag0)
    # win.setTitle('Diagonal %d' % k)
    # win.show()

    out = mpo.compute(x)
    # We compress it to easily compare with the GPU version
    out = np.hstack(out)

    # We test with half the windowLength requirement, which seems
    # like a good number
    mpoc = MatrixProfile(windowLength, True, 0.9, 0)
    chains = mpoc.compute(x)
    chains = np.array(chains)
    print(chains)
    # for chain in chains:
    #     if chain[1] - chain[0] == k:  # Just plotting over to look at it
    #         win.addItem(pg.InfiniteLine(chain[1]))
    #         win.addItem(pg.InfiniteLine(chain[1] + chain[2]))

    # Compute raw output
    cpmpo = CupyMatrixProfile(windowLength)
    d_out = cpmpo.compute(cp.asarray(x).astype(cp.complex64))

    compareValues(d_out[:out.size].get(), out)

    cpmpoc = CupyMatrixProfile(windowLength, True, 0.9, 0)
    print(cpmpoc._minThreshold)
    cchains = cpmpoc.compute(cp.asarray(x).astype(cp.complex64))
    cchains = np.array(cchains)
    print(cchains)

    # Unittest code
    # For float32s, 1e-5 rtol seems to be ok most of the time,
    # but you may get some edge case failures, so it's left at 1e-4
    # class TestMatrixProfile(unittest.TestCase):
    #     def setUp(self):
    #         self.totalLength = 100
    #         self.windowLength = 10
    #         self.numCopies = 3
    #         self.syms, self.bits = randPSKsyms(
    #             self.windowLength, 4, dtype=np.complex64)
    #         noise, self.x = addManySigToNoise(
    #             self.totalLength,
    #             np.arange(self.numCopies) * self.windowLength * 3,
    #             [self.syms for i in range(self.numCopies)],
    #             1, 1, [40.0 for i in range(self.numCopies)]
    #         )
    #         self.x = self.x.astype(np.complex64)
    #
    #         self.mpo = MatrixProfile(self.windowLength)
    #
    #     def test_computeNormsSq(self):
    #         normsSq = self.mpo._computeNormsSq(self.x)
    #         # Check that the length is correct
    #         self.assertEqual(
    #             normsSq.size,
    #             self.totalLength - self.windowLength + 1)
    #
    #         # Check all values are correct
    #         for i, nsq in enumerate(normsSq):
    #             np.testing.assert_allclose(
    #                 nsq,
    #                 np.linalg.norm(self.x[i:i+self.windowLength])**2,
    #                 rtol=1e-4
    #             )
    #
    #     def test_computeDiagonal(self):
    #         # Here we manually set normsSq for testing purposes
    #         self.mpo._normsSq = self.mpo._computeNormsSq(self.x)
    #         for k in range(1, self.totalLength - self.windowLength + 1):
    #             diag0 = self.mpo._computeDiagonal(self.x, k)
    #             # Check length of k-diagonal
    #             self.assertEqual(
    #                 diag0.size,
    #                 self.x.size - self.windowLength + 1 - k
    #             )
    #
    #             for i, dv in enumerate(diag0):
    #                 qf2 = calcQF2(
    #                     self.x[i:i+self.windowLength],
    #                     self.x[i+k:i+k+self.windowLength]
    #                 )
    #                 np.testing.assert_allclose(
    #                     dv, qf2,
    #                     rtol=1e-4
    #                 )
    #
    #     def test_compute(self):
    #         mp = self.mpo.compute(self.x)
    #         # Check each one manually
    #         for j, diag0 in enumerate(mp):
    #             k = j+1
    #             for i, dv in enumerate(diag0):
    #                 qf2 = calcQF2(
    #                     self.x[i:i+self.windowLength],
    #                     self.x[i+k:i+k+self.windowLength]
    #                 )
    #                 np.testing.assert_allclose(
    #                     dv, qf2,
    #                     rtol=1e-4
    #                 )
    #
    # class TestCupyMatrixProfile(unittest.TestCase):
    #     def setUp(self):
    #         self.totalLength = 100
    #         self.windowLength = 10
    #         self.numCopies = 3
    #         self.syms, self.bits = randPSKsyms(
    #             self.windowLength, 4, dtype=np.complex64)
    #         noise, self.x = addManySigToNoise(
    #             self.totalLength,
    #             np.arange(self.numCopies) * self.windowLength * 3,
    #             [self.syms for i in range(self.numCopies)],
    #             1, 1, [40.0 for i in range(self.numCopies)]
    #         )
    #         self.d_x = cp.asarray(self.x).astype(cp.complex64)
    #
    #         self.mpo = CupyMatrixProfile(self.windowLength)
    #
    #     def test_computeNormsSq(self):
    #         d_normsSq = self.mpo._computeNormsSq(self.d_x)
    #         normsSq = d_normsSq.get()
    #         # Check that the length is correct
    #         self.assertEqual(
    #             normsSq.size,
    #             self.totalLength - self.windowLength + 1)
    #
    #         # Check all values are correct
    #         for i, nsq in enumerate(normsSq):
    #             np.testing.assert_allclose(
    #                 nsq,
    #                 np.linalg.norm(self.x[i:i+self.windowLength])**2,
    #                 rtol=1e-4
    #             )
    #
    #     def test_computeDiagonal(self):
    #         # Here we manually set normsSq for testing purposes
    #         self.mpo._normsSq = self.mpo._computeNormsSq(self.d_x)
    #         for k in range(1, self.totalLength - self.windowLength + 1):
    #             d_diag0 = self.mpo._computeDiagonal(self.d_x, k)
    #             diag0 = d_diag0.get()
    #             # Check length of k-diagonal
    #             self.assertEqual(
    #                 diag0.size,
    #                 self.x.size - self.windowLength + 1 - k
    #             )
    #
    #             for i, dv in enumerate(diag0):
    #                 qf2 = calcQF2(
    #                     self.x[i:i+self.windowLength],
    #                     self.x[i+k:i+k+self.windowLength]
    #                 )
    #                 np.testing.assert_allclose(
    #                     dv, qf2,
    #                     rtol=1e-4
    #                 )
    #
    #     def test_compute(self):
    #         # This test should be identical, even though
    #         # the subclass doesn't reimplement.
    #         mp = self.mpo.compute(self.d_x)
    #         # Check each one manually
    #         for j, diag0 in enumerate(mp):
    #             k = j+1
    #             for i, dv in enumerate(diag0):
    #                 qf2 = calcQF2(
    #                     self.x[i:i+self.windowLength],
    #                     self.x[i+k:i+k+self.windowLength]
    #                 )
    #                 # Remember it's now a list of cupy arrays
    #                 np.testing.assert_allclose(
    #                     dv.get(), qf2,
    #                     rtol=1e-4
    #                 )
    #
    # unittest.main(verbosity=2)
