# Multi-Preamble Correlator

Here we discuss a possible implementation of a multi-preamble correlator. For simplicity, we consider only equal duration preambles (user is expected to truncate or pad zeroes accordingly).

The expected requirements are that this delivers a CAF with frequency resolution minimally equal to that of the preamble.

Input requirements:

1. Preambles $y^{i}$ are of equal length, and are delivered at a sample rate of $f_{s,p}$.
2. Input array $x$ (which is searched for the possible preambles) has a sample rate that is an integer multiple of the preambles' sample rate: $f_s = \eta f_{s,p}$.
3. Normalisation by the energy of both the respective preamble and the corresponding current time slice is performed: coefficient is $\frac{1}{|x[...]|^2/|y^{i}|^2}$

Some definitions:

1. $L$: sample length of each preamble.
2. $N$: number of time indices searched. This implies an input array length of at least $N+L-1$ if no elements are skipped.

## Method description

This method exploits the fact that a brick-wall upsampler to an integer multiple retains the frequency bins of the original vector as a subset of the new frequency bins.

TODO

## Comparison to sliding dot product + FFTs

Algorithm is briefly given by the following:

1. For each time index, cut a slice from $x$ of length $L$ and multiply it against a preamble.
2. Perform the FFT of this interrim array and save it.
3. Normalise the result.

Computational complexity for this in the asymptotic limit of large $N$ is determined solely by the FFTs, which gives $O(N L \log L)$.

## Comparison to generalised xcorr

Algorithm is briefly given by the following:

1. Perform a full FFT of the entire input $x$.
2. Perform an FFT of the zero-padded preamble, up to the same length as the first step.
3. Perform multiplies of the two FFTs, with up to $L$ circular shifts to be functionally equivalent to the frequency shift resolution of the preamble alone.
4. IFFT back; all FFTs and this IFFT are length $N$ minimally.
5. Normalise the results (this may require another iteration over the input array?).

Computational complexity is governed by the circular shift + IFFT steps, which are asymptotically $O(L N \log N)$. This is generally slower than the previous method, but offers the ability to select a frequency scanning range, so it may be preferred in some cases. Due to the nature of this setup, it also has finer frequency resolution.