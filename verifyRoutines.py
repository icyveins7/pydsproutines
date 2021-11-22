# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 14:08:28 2021

@author: Seo
"""

import numpy as np
import matplotlib.pyplot as plt

def compareValues(x: np.ndarray, y: np.ndarray, plotAbs=False):
    '''
    Helper routine to compare implementation outputs (usually a 64f vs 32f, or 64fc vs 32fc),
    for verification of reasonable accuracy.

    Parameters
    ----------
    x : np.ndarray
        Reference array (usually the higher precision array). This array is used for percentage amplitude checks.
    y : np.ndarray
        Test array (usually the lower precision array).

    Returns
    -------
    None.

    '''
    
    ii = np.argmax(np.abs(x-y))
    print("Values with largest raw change (index %d):" % (ii))
    print(x[ii])
    print(y[ii])
    
    nonzeros = np.argwhere(x!=0)
    x_nonzero = x[nonzeros]
    y_nonzero = y[nonzeros]
    iip = np.argmax(np.abs(x_nonzero-y_nonzero) / np.abs(x_nonzero))
    print("Values with largest %% change (index %d):" % (nonzeros[iip]))
    print(x_nonzero[iip])
    print(y_nonzero[iip])
    
    if plotAbs:
        plt.figure()
        plt.plot(np.abs(x), label='x')
        plt.plot(np.abs(y), label='y')
        plt.plot(np.abs(x-y), 'k--', label='x-y')
        plt.legend()
    