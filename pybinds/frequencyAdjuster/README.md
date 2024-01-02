# Goal

This should be the backend to a UI which allows for on-the-fly frequency adjustments to a constellation.

The user should ideally be able to drag a slider to increase/decrease the frequency offset, to get a feel for the correct value at which the constellation is well clustered.

Requirements:
1. Threaded implementation
2. Condition variables

# Input
For now, deal with complex64 i.e. 32-bit floating complex only.

# Implementation Details
Need to investigate which way might be better.
## Method 1: Convert Input to Magn/Phase internally

Given input array `x` we first split it into two contiguous parts `xMagn` and `xPhase`. The rotations are only concerned with `xPhase`.

During a change of frequency offset, we must do the following:

1. Write new phases to output array, based on an accumulating addition. This should be O(N), on a single real array.
2. Convert back to complex i.e. x/y values. This is also O(N) for both the `cos()` and `sin()` method calls, with 2 corresponding multiplications of the `xMagn` values. This will be expensive, since we invoke the `cos` and `sin` methods.

One good thing about this is that it naturally leaves the output in 2 separate arrays, which is usually how plotters take the input (as opposed to having to deinterleave the complex array).

## Method 2: Complex Multiplies

No initialization is necessary and we keep input `x` as it is.

During a change of frequency offset, we must do the following:

1. Calculate an accumulating-phase complex coefficient. This is O(N) in `cos()`, `sin()`. However, IPP alleviates this complexity by using some form of accumulator/differencing algorithm.
2. Multiply the coefficient with the input. Complex multiplies consist of 4 real multiplies and 2 additions, but this should be O(N).

## Initial Timings

### MacOS

Timings may be inaccurate due to cross-compile support for M1 + IPP. Tested with 1M elements.

Method 1: 12.4ms. <br>
Method 2:  5.2ms.

`CplxToReal` calls are insignificant, so for method 2 most of the computation time is in the `Tone` and `Mul` calls.

However for method 1 the `PolarToCartDeinterleaved` call is extremely significant. Disabling this call and the `CplxToReal` call in method 2 shows that the timing favours method 1 again.
