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

1. Write new phases to output array, based on an accumulating addition. This should be O(N).
2. Convert back to complex i.e. x/y values. This is also O(N) for both the `cos()` and `sin()` method calls, with 2 corresponding multiplications of the `xMagn` values.

One good thing about this is that it naturally leaves the output in 2 separate arrays, which is usually how plotters take the input (as opposed to having to deinterleave the complex array).

## Method 2: Complex Multiplies

No initialization is necessary and we keep input `x` as it is.

During a change of frequency offset, we must do the following:

1. Calculate an accumulating-phase complex coefficient. This is O(N) in `cos()`, `sin()`.
2. Multiply the coefficient with the input. Complex multiplies consist of 4 real multiplies and 2 additions, but this should be O(N).