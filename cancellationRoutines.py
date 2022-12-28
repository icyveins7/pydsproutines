#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 10:50:40 2022

@author: seolubuntu
"""

import numpy as np

def cancelSignalAtIdx(sig: np.ndarray, rx: np.ndarray, idx: int):
    '''
    Estimates the complex amplitude of a signal in a received vector, and then removes it.

    Parameters
    ----------
    sig : np.ndarray
        The target signal to remove (with appropriate frequency shifts already applied).
    rx : np.ndarray
        The received vector of samples.
    idx : int
        The starting sample of the signal in rx.

    Returns
    -------
    cancelled : np.ndarray
        The output vector after the signal removal.
    amp : complex float/double
        The estimated complex amplitude.
    '''
    # Estimate complex amplitude of signal
    pdt = np.vdot(sig, rx[idx:idx+sig.size])
    amp = pdt / np.linalg.norm(sig)**2
    
    # Create the cancelled version
    cancelled = rx.copy()
    cancelled[idx:idx+sig.size] -= amp * sig
    
    return cancelled, amp


if __name__ == "__main__":
    from signalCreationRoutines import *
    from xcorrRoutines import *
    
    syms1, bits = randPSKsyms(10000, 4)
    syms2, bits = randPSKsyms(10000, 4)
    _, sig = addSigToNoise(10000, 0, syms1+syms2*np.exp(1j*np.random.rand()*2*np.pi), snr_inband_linear=10.0)
    
    # Check QF2
    qf2 = calcQF2(syms1, sig)
    print("Syms 1")
    print(qf2)
    print("SNR = %f" % convertQF2toSNR(qf2))
    
    qf2 = calcQF2(syms2, sig)
    print("Syms 2")
    print(qf2)
    print("SNR = %f" % convertQF2toSNR(qf2))
    
    # Remove one
    cancelled, amp = cancelSignalAtIdx(syms2, sig, 0)
    qf2 = calcQF2(syms1, cancelled)
    print("Syms 1 after cancellation")
    print(qf2)
    print("SNR = %f" % convertQF2toSNR(qf2))
    