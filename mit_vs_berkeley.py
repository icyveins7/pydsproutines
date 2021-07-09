# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 12:53:16 2021

@author: User
"""

import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
import math
from averagingEllipsesRoutines import averageEllipses_Berkeley, averageEllipses_Davis, plotEllipse, pointInEllipse

#%% Sanity check

# ellipse_mu = np.zeros((2,2,1))
# ellipse_mu[0,:] = np.array([[0],[0]])
# ellipse_mu[1,:] = np.array([[5],[0]])
# ellipse_major = np.array([[3], [3]]) # one sigma semi-major 
# ellipse_minor = np.array([[1], [1]]) # one sigma semi-minor 
# ellipse_angle = np.array([[math.pi/2], [math.pi/2]]) # angle of semi-major w.r.t x-axis, anti-clockwise

# num_ellipse = np.shape(ellipse_mu)[0]

inEllipse_Davis = 0
inEllipse_Berkeley = 0
sim_num = 1
plot_flag = True

for k in np.arange(sim_num):
    
    # %% Generate ellipses 
    
    # # Ground truth ellipse
    # ref_mu = np.array([[0],[0]])
    
    # # Simulation ellipses parameters
    # num_ellipse = 6
    # mu_sigma = 2

    # major_mean = 2
    # major_sigma = 1
    # minor_mean = 0.5
    # minor_sigma = 0.25
    
    # ellipse_x = np.random.normal(ref_mu[0], mu_sigma, num_ellipse)
    # ellipse_y = np.random.normal(ref_mu[1], mu_sigma, num_ellipse)
    # ellipse_xy = np.vstack((ellipse_x,ellipse_y)).T.reshape(num_ellipse,2,1)
    # ellipse_mu = ellipse_xy.reshape(num_ellipse,2,1)
    
    # ellipse_major = np.random.normal(major_mean, major_sigma, num_ellipse).reshape(num_ellipse,1)
    # ellipse_minor = np.random.normal(minor_mean, minor_sigma, num_ellipse).reshape(num_ellipse,1)
    # ellipse_angle = (np.random.rand(num_ellipse,1)-0.5)*2*math.pi # angle sample from uniform distribution
    
    # %% Compute covariance matrices for generated ellipses
    
    # ellipse_cov = np.zeros((num_ellipse,2,2))
    
    # for n in np.arange(num_ellipse):

    #     ellipse_diag = np.array([[ellipse_major[n,0]**2, 0], [0, ellipse_minor[n,0]**2]])
    #     ellipse_rot = np.array([[np.cos(ellipse_angle[n,0]), -np.sin(ellipse_angle[n,0])], [np.sin(ellipse_angle[n,0]), np.cos(ellipse_angle[n,0])]]) # ellipse rotation matrix 
    #     ellipse_cov[n,:,:] = ellipse_rot@ellipse_diag@(ellipse_rot.T)
        
    #%% Generate distribution(test)
    
    # Ground truth distribution
    ref_mu = [0, 0]
    ref_cov = [[2, 0], [0, 2]]
    
    # Simulation ellipses parameters
    num_ellipse = 1000
    
    ellipse_x, ellipse_y = np.random.multivariate_normal(ref_mu, ref_cov, num_ellipse).T
    ellipse_xy = np.vstack((ellipse_x,ellipse_y)).T
    ellipse_mu = ellipse_xy.reshape(num_ellipse,2,1)
    ellipse_cov = np.zeros((num_ellipse,2,2))
    
    for n in np.arange(num_ellipse):
        ellipse_cov[n,:,:] = np.array(ref_cov)
    
    #%% Compute averaged ellipses
    
    mu_weightedMean_Davis, cov_Davis = averageEllipses_Davis(ellipse_mu, ellipse_cov)
    mu_weightedMean_Berkeley, cov_Berkeley = averageEllipses_Berkeley(ellipse_mu, ellipse_cov)
    
    # Plot ground truth
    if plot_flag:
        plt.plot(ref_mu[0], ref_mu[1], 'kx', label='Ground truth')
    
    n_sigma = 1;
    
    # Plot simulated ellipses
    for n in np.arange(num_ellipse):
        _, _, _, x, y = plotEllipse(ellipse_mu[n], ellipse_cov[n], n_sigma)
        if plot_flag:
            plt.plot(x,y,'b')
        
    # Plot Davis
    major_Davis, minor_Davis, angle_Davis, x, y = plotEllipse(mu_weightedMean_Davis, cov_Davis, n_sigma)
    if plot_flag:
        plt.plot(x,y,'r',label='Davis') 
    
    # Plot Berkeley
    major_Berkeley, minor_Berkeley, angle_Berkeley, x, y  = plotEllipse(mu_weightedMean_Berkeley, cov_Berkeley, n_sigma)
    if plot_flag:
        plt.plot(x,y,'m',label='Berkeley') 
    
    if plot_flag:
        plt.axis('square')
        plt.legend()
        plt.grid()
        
    if pointInEllipse(ref_mu, mu_weightedMean_Davis, major_Davis, minor_Davis, angle_Davis, n_sigma):
        inEllipse_Davis = inEllipse_Davis + 1
        print('Davis is in')
    
    if pointInEllipse(ref_mu, mu_weightedMean_Berkeley, major_Berkeley, minor_Berkeley, angle_Berkeley, n_sigma):
        inEllipse_Berkeley = inEllipse_Berkeley + 1
        print('Berkeley is in')

print('Theoretical %: ' + repr(chi2.cdf(n_sigma**2, 2)))
print('Davis %: ' + repr(inEllipse_Davis/sim_num))
print('Berkeley %: ' + repr(inEllipse_Berkeley/sim_num))






