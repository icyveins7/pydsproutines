#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 10:50:40 2022

@author: seolubuntu
"""

import numpy as np

def cancelSignalAtIdx(sig: np.ndadrray, rx: np.ndarray, idx: int):
    # Estimate complex amplitude of signal
    pdt = np.vdot(sig, rx[idx:idx+sig.size])
    amp = pdt / np.linalg.norm(sig)**2
    
    # Create the cancelled version
    cancelled = rx.copy()
    cancelled[idx:idx+sig.size] -= amp * sig
    
    return cancelled, amp