# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 12:24:22 2020

@author: Seo
"""

# closed, i don't think this is worth doing in pure phase angles, 
# the computations required to correct them in the loop are too ambiguous 
# and will require many steps, trumping the gains

import numpy as np
import scipy as sp
import scipy.signal as sps
import matplotlib.pyplot as plt

plt.close("all")

#%% initialization
numBits = 1000
m = 8

bits = np.random.randint(0,m,numBits)
syms = np.exp(1j*bits*2*np.pi/m)

#static phase offset
theta = -0.38
assert(theta > -(2*np.pi/m/2) and theta < (2*np.pi/m/2))
symsp = syms * np.exp(1j * theta)

#noise
noiseAmp = 0.1
symsn = symsp + ( np.random.randn(len(bits)) + np.random.randn(len(bits))*1j ) * noiseAmp

#freq offset
rFreq = 0.02
symsf = symsn * np.exp(1j*2*np.pi*rFreq*np.arange(len(symsn)))


plt.figure()
plt.subplot(2,2,1)
plt.plot(np.real(syms),np.imag(syms),'b.')
plt.axis("equal")
plt.title("Original")

plt.subplot(2,2,2)
plt.plot(np.real(symsp), np.imag(symsp), 'b.')
plt.axis("equal")
plt.title("Phase offset")

plt.subplot(2,2,3)
plt.plot(np.real(symsn),np.imag(symsn),'b.')
plt.axis("equal")
plt.title("Noise added")

plt.subplot(2,2,4)
plt.plot(np.real(symsf),np.imag(symsf),'b.')
plt.axis("equal")
plt.title("Noise + Freq shifted")

cm = symsn**m

plt.figure()
plt.plot(np.real(cm), np.imag(cm),'b.')
plt.title('CM')
plt.axis("equal")
plt.axis([-10,10,-10,10])

cmAvg = np.mean(cm)
theoPhase = np.angle(cmAvg)/m
print("Theoretical calc phase using CM = %f" %(theoPhase))




#%% attempt demod via angle metrics
magn = np.abs(symsf)
phases = np.angle(symsf)
phaseCorrection = -phases[0]
phasesC = phases + phaseCorrection

# note that the maximum search range should really be only until abs(rFreq) = +/- (1/m)/2, repeats every (1/m)
# freqSearchSpace = np.arange(-1/m/2,1/m/2,1/m/numBits/4)
freqSearchSpace = np.arange(-rFreq-0.001, -rFreq+0.001, 0.00001)
indicator = np.zeros(len(freqSearchSpace))
residualPhase = np.zeros(len(freqSearchSpace))

# put phase correction along with freq correction rather than a constant
for i in np.arange(len(freqSearchSpace)):
    # correct freq
    freqSearch = freqSearchSpace[i]
    phasesF = phases + 2 * np.pi * freqSearch * np.arange(len(phasesC))
        
    # estimate phase correction
    phasesM = phasesF * m
    
    aa_ori = np.remainder(phasesM,2*np.pi)
    bb_ori = np.array([i-2*np.pi if i > np.pi else i for i in aa_ori ])
    
    a1 = phasesM / (2*np.pi) # as an alternative to this, we do it in IPP specific function calls
    afract, aint = np.modf(a1)
    aa = afract * 2*np.pi
    
    bb = np.array([i-2*np.pi if i > np.pi else i for i in aa])
    bb = np.array([i+2*np.pi if i < -np.pi else i for i in bb])
    
    residualPhase[i] = np.mean(bb_ori)/m
    # residualPhase[i] = np.mean(bb)/m
    phasesCF = phasesF - residualPhase[i]
    
    # estimate bits
    estSyms = phasesCF / (2*np.pi/m)
    estBits = np.round(estSyms) # here we dont mod by m yet since we need to compare directly to the phases

    # estimate indicator
    estLockPhases = estBits * 2 *np.pi/m
    indicator[i] = np.linalg.norm(estLockPhases - phasesCF)
    
plt.figure()
plt.plot(freqSearchSpace, indicator)
plt.plot([-rFreq, -rFreq], [np.min(indicator), np.max(indicator)], 'r--')

#calculate the true theoretical indicator
pFT = phases + 2 * np.pi * -rFreq * np.arange(len(phasesC))
pCFT = pFT - theta
symsT = pCFT / (2*np.pi/m)
bitsT = np.round(symsT)
lockPhasesT = bitsT * 2 *np.pi/m
indicatorT = np.linalg.norm(lockPhasesT - pCFT)

# select the best one
bestIndicatorInd = np.argmin(indicator)
print("Residual phase = %f, correct phase = %f" % (residualPhase[bestIndicatorInd], theta))
selectPhases = phases + 2 * np.pi * freqSearchSpace[bestIndicatorInd] * np.arange(len(phasesC)) - residualPhase[bestIndicatorInd]
selectSyms = selectPhases / ( 2 * np.pi / m)
selectBits = (np.round(selectSyms) % m).astype(np.int32)
plt.figure(123)
plt.plot(magn * np.cos(selectPhases), magn * np.sin(selectPhases), 'b.')
# plot the phase separation lines
sepLineLen = np.max(magn)
for i in np.arange(m/2):
    a = (i + 0.5) * 2 * np.pi / m
    plt.plot(sepLineLen * np.array([np.cos(a),np.cos(a+np.pi)]), sepLineLen * np.array([np.sin(a), np.sin(a+np.pi)]), 'r--')
    
plt.axis("equal")
plt.title("Selected freq correction")

# for actual comparison, we move to the correct bits using the known ones
selectBitsComp = (selectBits + bits[0] - selectBits[0]) % m
if(not np.all(selectBitsComp == bits)):
    wrongIdx = np.argwhere(selectBitsComp != bits).flatten()
    print('%d wrong bits!\n' % (len(wrongIdx)))
    for i in range(len(wrongIdx)):
        wi = wrongIdx[i]
        plt.plot(magn[wi]*np.cos(selectPhases[wi]), magn[wi]*np.sin(selectPhases[wi]),'rx')
else:
    print('All bits correct by rough phase correction guess')
    
