# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 15:43:51 2020

@author: Seo
"""


# add the outside code routines
import sys
addedpaths = ["F:\\PycharmProjects\\pydsproutines"]
for path in addedpaths:
    if path not in sys.path:
        sys.path.append(path)

# imports
import numpy as np
import scipy as sp
import scipy.signal as sps
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from signalCreationRoutines import *
from xcorrRoutines import *
from filterCreationRoutines import *
from demodulationRoutines import *
from pgplotRoutines import *
from PyQt5.QtWidgets import QApplication
import time
import matplotlib.pyplot as plt
# end of imports

QApplication.closeAllWindows()

#%% parameters
numBitsPerBurst = 48
baud = 16000
numBursts = 20
numBitsTotal = numBitsPerBurst * numBursts
m = 2 # m-ary
h = 1.0/m
up = 8
print('Duration of burst = %fs' % (numBitsTotal/baud))

# create bits
bits = randBits(numBitsTotal, m)

# create cpfsk signal
gflat = np.ones(up)/(2*up)

# create SRC4 CPFSK symbols
gSRC4 = makeSRC4(np.arange(4 * up)/up,1)
gSRC4 = makeScaledSRC4(up, 1.0)/up
syms0, fs, data = makePulsedCPFSKsyms(bits, baud, g = gSRC4, up = up) # new method of creation
T = 1/fs
print('\nWorking at fs = %fHz, sample period T = %gs' % (fs, T))

#%% test with noise, arbitrary frequency shift and 


    
    