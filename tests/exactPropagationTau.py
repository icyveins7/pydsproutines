# -*- coding: utf-8 -*-
"""
Created on Mon May 15 14:58:33 2023

@author: Seo
"""

from trajectoryRoutines import *
from plotRoutines import *
closeAllFigs()

import numpy as np

from scipy.constants import speed_of_light as lightspd

#%%
traj = ConstantVelocityTrajectory(np.array([3e7,0,0]), np.array([300, 0, 0]))

# Assume tx at origin
tx_x = np.zeros(3)

# First guess
tauhat = np.linalg.norm(tx_x - traj.at(0)) / lightspd
print("First guess tau = %.15g" % tauhat)
# Check guess
check = lightspd * tauhat - np.linalg.norm(tx_x - traj.at(tauhat))
print("Minimized check = %.15g (metres)" % check)
# Iterate
for i in range(3):
    tauhat = np.linalg.norm(tx_x - traj.at(tauhat)) / lightspd
    print("tauhat = %.15g" % tauhat)
    
check = lightspd * tauhat - np.linalg.norm(tx_x - traj.at(tauhat))
print("Minimized check = %.15g (metres)" % check)
    
#%% Test using velocity instead
from numpy.polynomial import Polynomial

v = traj.v
D = tx_x - traj.x0

p = Polynomial((np.linalg.norm(D)**2, -2*np.dot(v,D), (-lightspd**2 - np.linalg.norm(v)**2)))
# print(p.roots())
tauhatpoly = p.roots()[1]

print("Tau estimation using velocity = %.15g" % tauhatpoly)
polycheck = lightspd * tauhatpoly - np.linalg.norm(tx_x - traj.at(tauhatpoly))
print("Minimized check = %.15g (metres)" % polycheck)
