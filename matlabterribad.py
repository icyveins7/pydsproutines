# -*- coding: utf-8 -*-
"""
Created on Wed May 24 16:55:19 2017

@author: LKen
"""

import numpy as np
import matplotlib.pyplot as plt

f = open('D:\Amos4\originalsig\USRP_int16_10s_AWGN_400k.bin','rb')
data = np.fromfile(f,np.int16)
f.close()
x_raw = np.zeros([np.shape(data)[0]/2,1],dtype = complex)
for i in range(np.size(x_raw)):
    x_raw[i] = data[i*2] + 1j*data[i*2+1]

fftx = np.fft.fft(x_raw,axis=0)

plt.plot(fftx[0:400000])