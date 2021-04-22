# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 18:14:31 2021

@author: Seo
"""

import numpy as np
from cztRoutines import *
import matplotlib.pyplot as plt

fs = 192000.0

fstep = 0.01
f = np.arange(-10,10+0.005,fstep)

tonef = 1000
phi = np.pi/4
length = fs * 1
tone = np.exp(1j*(2*np.pi*tonef*np.arange(length)/fs + phi))
N = len(tone)
tau = N/fs

cc = czt(tone, np.round(tonef+f[0]), np.round(tonef+f[-1]), fstep, fs)

sincfunc = np.sinc(f * tau)

plt.figure()
plt.plot(f+tonef, np.abs(cc))
plt.plot(f+tonef, N * sincfunc)

peakIdx = int(np.median(np.arange(f.size)))
phaseAtPeak = np.angle(cc[peakIdx])

plt.figure()
plt.plot(f+tonef,np.angle(cc))
plt.plot(f+tonef, np.angle(N*sincfunc))

dd = czt(np.ones(tone.size), np.round(f[0]), np.round(f[-1]), fstep, fs)
dd = dd * np.exp(1j*phi)

plt.plot(f+tonef, np.angle(dd), 'r--')
