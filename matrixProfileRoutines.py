import numpy as np


class MatrixProfile:
    def __init__(
        self,
        windowLength: int = 10,
        onlyUpperTriangle: bool = True
    ):
        self._windowLength = windowLength
        self._norms = None

    def compute(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the matrix profile.

        Parameters
        ----------
        x : np.ndarray
            The time series.

        Returns
        -------
        mp : np.ndarray
            The matrix profile.
        """
        self._computeNorms(x)

        raise NotImplementedError()

    def _computeNorms(self, x: np.ndarray) -> np.ndarray:
        power = np.abs(x)**2
        self._norms = np.convolve(
            ampl, np.ones(self._windowLength), mode='valid')

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

        pdt = slice1 * slice2.conj()
        kdiag = np.convolve(pdt, np.ones(self._windowLength), mode='valid')
        return kdiag


if __name__ == "__main__":
    from plotRoutines import *
    from signalCreationRoutines import randPSKsyms, addManySigToNoise

    syms, bits = randPSKsyms(10, 4, dtype=np.complex64)
    noise, x = addManySigToNoise(
        500,
        np.arange(5) * 20,
        [syms for i in range(5)],
        1, 1, [10 for i in range(5)]
    )

    mpo = MatrixProfile()
    diag0 = mpo._computeDiagonal(x, 1)
    print(diag0)
    print(diag0.shape)

    pgPlotAmpTime(diag0)
