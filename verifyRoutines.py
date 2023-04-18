# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 14:08:28 2021

@author: Seo
"""

import numpy as np
import matplotlib.pyplot as plt

def compareValues(x: np.ndarray, y: np.ndarray, plotAbs: bool=False, verbose: bool=True):
    '''
    Helper routine to compare implementation outputs (usually a 64f vs 32f, or 64fc vs 32fc),
    for verification of reasonable accuracy.

    Parameters
    ----------
    x : np.ndarray
        Reference array (usually the higher precision array). This array is used for percentage amplitude checks.
    y : np.ndarray
        Test array (usually the lower precision array).
    plotAbs : bool, optional
        Plots a simple comparison of the absolute values and differences of the 2 arrays. The default is False.
    verbose : bool, optional
        Prints the worst raw and fractional change values. The default is True.

    Returns
    -------
    rawChg : float
        Largest absolute change (corresponding to the first index printed).
        
    fracChg : float
        Largest fractional change (corresponding to the second index printed).

    '''
    
    ii = np.argmax(np.abs(x-y))
    if verbose:
        print("Values with largest raw change (index %d):" % (ii))
        print(x[ii])
        print(y[ii])
    rawChg = np.abs(x[ii]-y[ii])
    
    nonzeros = np.argwhere(x!=0)
    x_nonzero = x[nonzeros]
    y_nonzero = y[nonzeros]
    iip = np.argmax(np.abs(x_nonzero-y_nonzero) / np.abs(x_nonzero))
    if verbose:
        print("Values with largest %% change (index %d):" % (nonzeros[iip]))
        print(x_nonzero[iip])
        print(y_nonzero[iip])
    fracChg = np.abs(x_nonzero[iip]-y_nonzero[iip]) / np.abs(x_nonzero[iip])
    
    if plotAbs:
        plt.figure()
        plt.plot(np.abs(x), label='x')
        plt.plot(np.abs(y), label='y')
        plt.plot(np.abs(x-y), 'k--', label='x-y')
        plt.legend()

    return rawChg, fracChg
    
    