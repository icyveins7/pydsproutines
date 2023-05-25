# -*- coding: utf-8 -*-
"""
Created on Mon May 15 14:58:33 2023

@author: Seo
"""

from trajectoryRoutines import *
from plotRoutines import *
closeAllFigs()

import numpy as np

lightspd = 299792458.0

numSamp = 1000000
T = 1e-6
v = 1000.0

rx_x, rx_xdot = createLinearTrajectory(numSamp, np.array([10000e3, 0, 0]), np.array([1e18, 0, 0]), v, T)
t = np.arange(rx_x.shape[0])*T
rx = Receiver(rx_x, rx_xdot, t)

tx = Transmitter.asStationary(np.zeros((numSamp,3)), t)

tauhats = np.linalg.norm(rx.x - tx.x, axis=1) / lightspd

def taucost(tx, rx, tau, t0):
    # Assume 1D, only x is relevant, ignore y and z
    if t0+tau > rx.t[-1]:
        raise ValueError("Out of bounds")
    
    return lightspd * tau - np.abs(tx.x[0,0] - np.interp(t0+tau, rx.t, rx.x[:,0]))

for i, tauhat in enumerate(tauhats):
    tauActual = np.linalg.norm(rx.x[0]) / (lightspd - rx.xdot[0,0])
    
    tc = taucost(tx, rx, tauhat, t[i])
    tauhatp = tauhat
    
    k = 0
    while np.abs(tc) > 1e-6:
        
        # if tc < 0:
        tauhatp = np.linalg.norm(tx.x[0,0] - np.interp(t[i]+tauhatp, rx.t, rx.x[:,0])) / lightspd
        # else:
        #     tauhatp = np.linalg.norm(tx.x[0,0] - np.interp(t[i]-tauhatp, rx.t, rx.x[:,0])) / lightspd
        
        # if tc < 0:
        #     tauhatp += 1e-9
        # else:
        #     tauhatp -= 1e-9
        
        newtc = taucost(tx, rx, tauhatp, t[i])
        if k > 0 and np.sign(newtc) != np.sign(tc):
            print("exit")
            break
        tc = newtc
        k += 1
        print(newtc, tauhatp)
    
    
    print("First", tauActual - tauhat, (tauActual-tauhat)/tauActual)
    print("Refined", tauActual - tauhatp, (tauActual-tauhatp)/tauActual)
    
    
    break

# plt.plot(tauhat)