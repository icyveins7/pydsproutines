# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 15:36:35 2021

@author: Perry Hong
"""

import numpy as np
import math

#%% Averaging functions

def averageEllipses_Davis(ellipse_mu, ellipse_cov):

    num_ellipse = np.shape(ellipse_mu)[0]
    
    ellipse_cov_inv_sum = np.zeros((2,2))
    for n in np.arange(num_ellipse):
        ellipse_cov_inv_sum = ellipse_cov_inv_sum + np.linalg.inv(ellipse_cov[n,:,:]) # unnormalised 
        
    cov_Davis = np.linalg.inv(ellipse_cov_inv_sum)
    
    mu_weightedMean = np.zeros((2,1))
    for n in np.arange(num_ellipse):
        mu_weightedMean = mu_weightedMean + np.linalg.inv(ellipse_cov[n,:,:])@ellipse_mu[n] # inverse-variance weighted mean
        
    mu_weightedMean = cov_Davis@mu_weightedMean# normalisation of inverse-variance weightage

    return mu_weightedMean, cov_Davis
    
def averageEllipses_Berkeley(ellipse_mu, ellipse_cov):
    
    num_ellipse = np.shape(ellipse_mu)[0]
    
    # initial computation exactly that of Davis, as weighted mean still utilises inverse-variance weightage
    ellipse_cov_inv_sum = np.zeros((2,2))
    for n in np.arange(num_ellipse):
        ellipse_cov_inv_sum = ellipse_cov_inv_sum + np.linalg.inv(ellipse_cov[n,:,:]) # unnormalised 
        
    cov_Davis = np.linalg.inv(ellipse_cov_inv_sum)
    
    mu_weightedMean = np.zeros((2,1))
    for n in np.arange(num_ellipse):
        mu_weightedMean = mu_weightedMean + np.linalg.inv(ellipse_cov[n,:,:])@ellipse_mu[n] # inverse-variance weighted mean
        
    mu_weightedMean = cov_Davis@mu_weightedMean # normalisation of inverse-variance weightage
    
    numerator = np.zeros((2,2))
    for n in np.arange(num_ellipse):
        ellipse_weight = cov_Davis @ np.linalg.inv(ellipse_cov[n,:,:]) # normalised inverse-variance weightage
        numerator = numerator + ellipse_weight * ((ellipse_mu[n]-mu_weightedMean).T @ (ellipse_mu[n]-mu_weightedMean))
    
    cov_Berkeley = numerator*num_ellipse/(num_ellipse-1)/num_ellipse
    
    return mu_weightedMean, cov_Berkeley

#%% Plotting function
# n_sigma = number of sigma to plot

def plotEllipse(mu, cov, n_sigma):
    
    rot, diag, _ = np.linalg.svd(cov, full_matrices=True)
    
    # Convert to one sigma semi-major, semi-minor, and angle semi-major makes w.r.t. x-axis
    major = np.sqrt(diag[0])
    minor = np.sqrt(diag[1])
    angle = np.arctan2(rot[1,0],rot[1,1])
    
    t = np.arange(0,2*math.pi,0.01)
    
    # Parametric equation for an ellipse
    x = major*n_sigma*np.cos(t)*np.cos(angle) - minor*n_sigma*np.sin(t)*np.sin(angle) + mu[0]
    y = major*n_sigma*np.cos(t)*np.sin(angle) + minor*n_sigma*np.sin(t)*np.cos(angle) + mu[1]
    
    return major, minor, angle, x, y

#%% Given ellipse major

def pointInEllipse(point, mu, major, minor, angle, n_sigma):
    
    inequality = ((np.cos(angle)*(point[0]-mu[0]) + np.sin(angle)*(point[1]-mu[1]))**2)/((major*n_sigma)**2) + \
        ((np.sin(angle)*(point[0]-mu[0]) - np.cos(angle)*(point[1]-mu[1]))**2)/((minor*n_sigma)**2)
    
    return inequality < 1
    

    
    
    
    
    
    