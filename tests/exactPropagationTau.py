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
for i in range(2):
    print("Iteration %d" % i)
    tauhat = np.linalg.norm(tx_x - traj.at(tauhat)) / lightspd
    print("tauhat = %.15g" % tauhat)
    
    check = lightspd * tauhat - np.linalg.norm(tx_x - traj.at(tauhat))
    print("Minimized check = %.15g (metres)" % check)
    
#%% Test using velocity instead
from numpy.polynomial import Polynomial

v = traj.v
D = tx_x - traj.x0

p = Polynomial((np.linalg.norm(D)**2, -2*np.dot(v,D), (np.linalg.norm(v)**2-lightspd**2)))
# print(p.roots())
tauhatpoly = p.roots()[1]

print("Tau estimation using velocity (Numpy Polynomial) = %.15g" % tauhatpoly)
polycheck = lightspd * tauhatpoly - np.linalg.norm(tx_x - traj.at(tauhatpoly))
print("Minimized check = %.15g (metres)" % polycheck)

# Check using 1D radial assumption
tauhat1d = np.linalg.norm(tx_x - traj.at(0))/(lightspd - np.linalg.norm(v))
print("Tau estimation using 1D radial assumption = %.15g" % (tauhat1d))

# Check using direct quadratic formula
a = (np.linalg.norm(v)**2-lightspd**2)
b = -2*np.dot(v,D)
c = np.linalg.norm(D)**2
tauhatquad = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
print("Tau estimation using direct quadratic formula = %.15g" % tauhatquad)
