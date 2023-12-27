# Multi-Preamble Correlator

Here we discuss a possible implementation of a multi-preamble correlator. For simplicity, we consider only equal duration preambles (user is expected to truncate or pad zeroes accordingly).

The expected requirements are that this delivers a CAF with frequency resolution minimally equal to that of the preamble.

Input requirements:

1. Preambles $y^{i}$ are of equal length, and are delivered at a sample rate of $f_{s,p}$.
2. Input array $x$ (which is searched for the possible preambles) has a oversampled rate that is an integer multiple of the preambles' sample rate: $f_s = \eta f_{s,p}$.
3. Normalisation by the energy of both the respective preamble and the corresponding current time slice is performed: coefficient is $\frac{1}{|x[...]|^2/|y^{i}|^2}$

Some definitions:

1. $L$: sample length of each preamble.
2. $N$: number of time indices searched. This implies an input array length of at least $N+L-1$ if no elements are skipped.

## Method description

This method exploits the fact that a brick-wall upsampler to an integer multiple retains the frequency bins of the original vector as a subset of the new frequency bins.

Consider a length $L=4$ preamble; the frequency bins are by definition at $[-\frac{1}{2}f_{s,p}, -\frac{1}{4}f_{s,p}, 0, \frac{1}{4}f_{s,p}]$.

Now consider an oversampling rate $\eta = 2$; the 8 frequency bins are by definition at $[-f_{s,p}, -\frac{3}{4}f_{s,p}, -\frac{1}{2}f_{s,p}, -\frac{1}{4}f_{s,p}, 0, \frac{1}{4}f_{s,p}, \frac{1}{2}f_{s,p}, \frac{3}{4}f_{s,p}]$. 

Example:

Original = $[a,b,c,d]$

Oversampled = $[0,0,a,b,c,d,0,0]$

It should be clear that the original frequency bins are strictly a __contiguous__ subset (no in between frequency bins) of the oversampled frequency bins.

Since correlation in the frequency domain is performed via multiplication, and a brick-wall filter results in 0-valued bins, then it suffices that the FFT of the smaller, critically-sampled $L=4$ preamble be performed, but multiplied (at specific circularly shifted indices) with the larger length $4\eta = 8$ FFT, extracted from the input.

Example:

Input FFT: $[x_0, ..., x_7]$

1st product: $[a,b,c,d] \times [x_0, x_1, x_2, x_3]$

2nd product: $[a,b,c,d] \times [x_1, x_2, x_3, x_4]$

...

5th (last) product: $[a,b,c,d] \times [x_4, x_5, x_6, x_7]$

This would have been functionally equivalent to brick-wall upsampling the original preamble by $\eta$, then performing circular shift multiplies of the FFT with an input FFT.

Note: it is important that the FFTs are fft-shifted (to a physically increasing frequency order). The legitimate number of products is then given by $\eta L - L + 1 = (\eta-1)L + 1$.

At this juncture, we deviate from the _normal_ correlation algorithm (which performs an IFFT). Instead we simply perform the sum (the zero bin of the IFFT). This is because we have not pre-zero-padded either of the original FFTs, so there is no room for the correlation to produce the 'sliding window' effect. Instead, we have to perform the above computations for each individual time index.

Computational complexity is given as follows:

1. Assume FFT(s) of preamble(s) are pre-computed and ignored.
2. For each of the $N$ searched time indices, perform length $\eta L$ FFT $\rightarrow O(\eta L \log \eta L)$
3. For each of the $N$ searched time indices, perform up to $(\eta-1)L + 1$ multiplies (or dot product, rather) of length $L$ $\rightarrow O((\eta-1)L^2)$
4. Perform some normalisation required.

Total complexity is

$$
O(N (\eta L \log (\eta L) + (\eta-1)L^2)) \approx O(N \eta L \log (\eta L) + N \eta L^2)
$$

However, this isn't the full story! We consider the case where multiple different preambles are computed; in these scenarios, we don't repeat the FFT for each time index. Instead, we only re-do the dot products. So for $K$ different preambles, the total complexity is 

$$
O(N \eta L \log (\eta L) + KN \eta L^2)
$$

Note that the FFT term is unscaled by $K$!

An easier way to consider this is to boil this down to a per-time-index complexity of 

$$
O(\eta L \log (\eta L) + K \eta L^2)
$$

## Comparison to sliding dot product + FFTs

Algorithm is briefly given by the following:

1. For each time index, cut a slice from $x$ of length $\eta L$ and multiply it against a preamble (here we use $\eta L$ to compare since this method necessitates pre-upsampling)
2. Perform the FFT of this interrim array and save it.
3. Normalise the result.

Computational complexity for this for a single preamble is given by $O(\eta L \log (\eta L) + \eta L)$.

However, for $K$ different preambles, this scales linearly as $O(K \eta L \log (\eta L) + K \eta L)$. The bad thing about this is that the FFT term is scaled along with the linear multiplication term!

If we compare the dominant term for this (the FFT term) with the dominant term for the multi-preamble method above (the $L^2$ term), then we can see that the comparison is

$$
O(L) \,\, \text{vs} \,\, O(\log (\eta L))
$$

### Complexity estimation examples

$\log_2$ is used where $\log$ is required. We compute scalar values by plugging in the full complexity formulas for both methods given above.

1. $\eta = 10, L = 10$. Expectation in the limit is that $\frac{L}{\log (\eta L)} \approx 1.5$, so the multi-preamble method should be worse by about 50%.

$K=1$.

Multi-preamble: $1664.39$.<br>
Sliding + FFT: $764.39$.

$K = 32$

Multi-preamble: $32664$.<br>
Sliding + FFT: $24460$.

2. $\eta = 1000, L = 10$. Expectation in the limit is that $\frac{L}{\log (\eta L)} \approx 0.75$, so the multi-preamble method should be better by about 25%.

$K=1$.

Multi-preamble: $232877$.<br>
Sliding + FFT: $142877$.

$K = 32$

Multi-preamble: $3332877$.<br>
Sliding + FFT: $4572067$.



## Comparison to generalised xcorr

Algorithm is briefly given by the following:

1. Perform a full FFT of the entire input $x$.
2. Perform an FFT of the zero-padded preamble, up to the same length as the first step.
3. Perform multiplies of the two FFTs, with up to $L$ circular shifts to be functionally equivalent to the frequency shift resolution of the preamble alone.
4. IFFT back; all FFTs and this IFFT are length $N$ minimally.
5. Normalise the results (this may require another iteration over the input array?).

Computational complexity is governed by the circular shift + IFFT steps, which are asymptotically $O(L N \log N)$. This is generally slower than the previous method, but offers the ability to select a frequency scanning range, so it may be preferred in some cases. Due to the nature of this setup, it also has finer frequency resolution.