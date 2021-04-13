# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 17:06:03 2021

@author: Lken
"""

import numpy as np
import scipy as sp
import time
from concurrent.futures import ThreadPoolExecutor


def burstFFT(x, length):
    # first check if length is a multiple of the original length
    if len(x) % length != 0:
        alpha = int(np.ceil(len(x)/length))
        N = alpha * length
        
    else:
        alpha = int(len(x)/length)
        N = len(x)
     
   # overlap add
    xp = np.pad(x, [0, N-len(x)])
    x_rs = xp.reshape((-1,length))
    
    t1 = time.time()
    x_rs_sum = np.sum(x_rs, axis=0)
    t2 = time.time()
    # fft at the end!
    out = np.fft.fft(x_rs_sum)
    t3 = time.time()
    # print("Took %f s for sum\nTook %f s for fft." % (t2-t1,t3-t2))
    
    return out

#%% deprecated
# def burstFFT_thread(t, x, startIdxs, length, alpha, N, mini_out, NUM_THREADS):
#     for i in np.arange(t, len(startIdxs), NUM_THREADS):
#         xs = x[startIdxs[i]:startIdxs[i] + length]
        
#         xsfft = np.fft.fft(xs)
        
#         tone = np.exp(-1j*2*np.pi*alpha*startIdxs[i]/N * np.arange(length))
        
#         mini_out[t] = mini_out[t] + xsfft * tone


# def burstFFT_dep(x, startIdxs, length, NUM_THREADS=4, verb=False):
#     # first check if length is a multiple of the original length
#     if len(x) % length != 0:
#         alpha = int(np.ceil(len(x)/length))
#         N = alpha * length
        
#     else:
#         alpha = int(len(x)/length)
#         N = len(x)
     

#     # allocate output
#     out = np.zeros(length,dtype=x.dtype)
#     mini_out = np.zeros((NUM_THREADS, length), dtype=x.dtype)

#     # use threads
#     with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
#         for t in range(NUM_THREADS):
#             executor.submit(burstFFT_thread, t, x, startIdxs, length, alpha, N, mini_out, NUM_THREADS)
#     # sum over mini_out
#     out = np.sum(mini_out,axis=0)
    
#     return out
