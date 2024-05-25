import numpy as np


class MatrixProfile:
    def __init__(
        self,
        windowLength: int,
    ):
        self._windowLength = windowLength
        self._normsSq = None

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

        # Loop over all diagonals
        mp = list()
        for i in range(1, x.size - self._windowLength + 1):
            # Compute the diagonal
            diag = self._computeDiagonal(x, i)
            mp.append(diag)

        return mp

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


if __name__ == "__main__":
    # The following are used for unittests and signal gen
    import unittest
    from xcorrRoutines import calcQF2
    from signalCreationRoutines import randPSKsyms, addManySigToNoise

    # These are used for example code and plotting only
    import pyqtgraph as pg
    from plotRoutines import pgPlotAmpTime, closeAllFigs
    closeAllFigs()
    #
    # windowLength = 10
    # numCopies = 3
    # syms, bits = randPSKsyms(windowLength, 4, dtype=np.complex64)
    # noise, x = addManySigToNoise(
    #     200,
    #     np.arange(numCopies) * windowLength * 3,
    #     [syms for i in range(numCopies)],
    #     1, 1, [40.0 for i in range(numCopies)]
    # )
    #
    # mpo = MatrixProfile(windowLength)
    # mpo._normsSq = mpo._computeNormsSq(x)
    # k = 30
    # diag0 = mpo._computeDiagonal(x, k)
    # print(diag0)
    # print(diag0.shape)
    #
    # awin, aax = pgPlotAmpTime(x, title='ampl-time')
    #
    # win = pg.plot(np.arange(diag0.size) + k, diag0)
    # win.setTitle('Diagonal %d' % k)
    # win.show()

    # Unittest code
    class TestMatrixProfile(unittest.TestCase):
        def setUp(self):
            self.totalLength = 100
            self.windowLength = 10
            self.numCopies = 3
            self.syms, self.bits = randPSKsyms(
                self.windowLength, 4, dtype=np.complex64)
            noise, self.x = addManySigToNoise(
                self.totalLength,
                np.arange(self.numCopies) * self.windowLength * 3,
                [self.syms for i in range(self.numCopies)],
                1, 1, [40.0 for i in range(self.numCopies)]
            )

            self.mpo = MatrixProfile(self.windowLength)

        def test_computeNormsSq(self):
            normsSq = self.mpo._computeNormsSq(self.x)
            # Check that the length is correct
            self.assertEqual(
                normsSq.size,
                self.totalLength - self.windowLength + 1)

            # Check all values are correct
            for i, nsq in enumerate(normsSq):
                np.testing.assert_allclose(
                    nsq,
                    np.linalg.norm(self.x[i:i+self.windowLength])**2
                )

        def test_computeDiagonal(self):
            # Here we manually set normsSq for testing purposes
            self.mpo._normsSq = self.mpo._computeNormsSq(self.x)
            for k in range(1, self.totalLength - self.windowLength + 1):
                diag0 = self.mpo._computeDiagonal(self.x, k)
                # Check length of k-diagonal
                self.assertEqual(
                    diag0.size,
                    self.x.size - self.windowLength + 1 - k
                )

                for i, dv in enumerate(diag0):
                    qf2 = calcQF2(
                        self.x[i:i+self.windowLength],
                        self.x[i+k:i+k+self.windowLength]
                    )
                    np.testing.assert_allclose(
                        dv, qf2
                    )

        def test_compute(self):
            mp = self.mpo.compute(self.x)
            # Check each one manually
            for j, diag0 in enumerate(mp):
                k = j+1
                for i, dv in enumerate(diag0):
                    qf2 = calcQF2(
                        self.x[i:i+self.windowLength],
                        self.x[i+k:i+k+self.windowLength]
                    )
                    np.testing.assert_allclose(
                        dv, qf2
                    )

    unittest.main()
