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
    from plotRoutines import *
    closeAllFigs()
    
    length = 100000
    syms1, bits = randPSKsyms(length, 4)
    syms2, bits = randPSKsyms(length, 4)
    _, sig = addSigToNoise(length, 0, syms1+syms2*np.exp(1j*np.random.rand()*2*np.pi), snr_inband_linear=10.0)
    
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
    
    # Add acceleration effects
    noise = randnoise(syms1.size, 1.0, 1.0, 10.0)
    # acc = 1e-10 # Above 1e-11 is when it starts to show effects, for this length
    # Technically since this is itself dependent on time, then it should be scaled according to length
    factor = 1.0 # This seems like a decent way to quantify? i.e. same shape regardless of length
    acc = 1/ (length/factor)**2 # 1.0 factor is when it seems to start to happen
    syms1acc = syms1 * np.exp(1j*2*np.pi*0.5*acc*np.arange(syms1.size)**2)
    sigacc = syms1acc + noise
    cancelled, amp = cancelSignalAtIdx(syms1, sigacc, 0)
    qf2 = calcQF2(syms1, sigacc)
    print("Syms 1 (with accel) before/after cancellation")
    print(qf2, calcQF2(syms1,cancelled))
    print("SNR = %f" % convertQF2toSNR(qf2))
    win, ax = pgPlotAmpTime([sigacc, cancelled], [1,1], ['orig','cancelled'], ['w','r'])
    