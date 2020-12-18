# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 11:59:47 2020

@author: Seo
"""


import numpy as np


def createLinearTrajectory(pos1, pos2, stepArray, pos_start=None, randomWalk=None):
    '''
    Parameters
    ----------
    pos1 : Numpy array, 1-D.
        Anchor position 1. Starting position defaults to this.
    pos2 : Numpy array, 1-D.
        Anchor position 2.
    stepArray : Numpy array, 1-D.
        Array of steps. Each point will move by step * directionVectorNormed. Does not need to be equally spaced.
    pos_start : Scalar between [0,1], optional
        The default is None.
        The output will use this as the coefficient along the connecting vector between
        the 2 anchor points, as the position to start iterating at.
    randomWalk : Scalar, optional
        The default is None.
        Adds random noise around the trajectory using a normal distribution.

    Returns
    -------
    Matrix of column vectors of positions along the trajectory.
    For steps which exceed the 2nd anchor point, the direction reverses i.e.
    the trajectory is constructed as a bounce between the two anchor points.
    '''
    
    if pos_start is None:
        pos0 = pos1
    else:
        raise NotImplementedError
        
    result = np.zeros((len(pos1), len(stepArray)))
    finalStepArray = np.zeros(stepArray.shape)
    
    dirVec = pos2 - pos1
    anchorDist = np.linalg.norm(dirVec)
    dirVecNormed = dirVec / np.linalg.norm(dirVec)
    # revDirVecNormed = -dirVecNormed
    
    lengthsCovered = np.floor(stepArray / anchorDist)
    idxReverse = np.argwhere(lengthsCovered%2==1).flatten()
    
    # in these indices, calculate the remaining length
    remainderLen = np.remainder(stepArray[idxReverse], anchorDist)
    
    # these are removed from the full length to induce the backward motion from the 2nd anchor point
    finalStepArray[idxReverse] = anchorDist - remainderLen 
    
    # for forward we do the same
    idxForward = np.argwhere(lengthsCovered%2==0).flatten()
    
    remainderForwardLen = np.remainder(stepArray[idxForward], anchorDist)
    
    finalStepArray[idxForward] = remainderForwardLen
    
    # now calculate the values
    displacements = dirVecNormed.reshape((-1,1))  * finalStepArray.reshape((1,-1))
    
    result = pos0.reshape((-1,1)) + displacements
    
    return result

def createCircularTrajectory(totalSamples, r_a=100000.0, desiredSpeed=100.0, r_h=300.0, sampleTime=3.90625e-6):    
    # initialize a bunch of rx points in a circle in 3d
    dtheta_per_s = desiredSpeed/r_a # rad/s
    arcangle = totalSamples * sampleTime * dtheta_per_s # rad
    r_theta = np.arange(0,arcangle,dtheta_per_s * sampleTime)
    
    r_x_x = r_a * np.cos(r_theta)
    r_x_y = r_a * np.sin(r_theta)
    r_x_z = np.zeros(len(r_theta)) + r_h
    r_x = np.vstack((r_x_x,r_x_y,r_x_z)).transpose()
    
    r_xdot_x = r_a * -np.sin(r_theta) * dtheta_per_s
    r_xdot_y = r_a * np.cos(r_theta) * dtheta_per_s
    r_xdot_z = np.zeros(len(r_theta))
    r_xdot = np.vstack((r_xdot_x,r_xdot_y,r_xdot_z)).transpose()
    
    return r_x, r_xdot, arcangle, dtheta_per_s

def calcFOA(r_x, r_xdot, t_x, t_xdot, freq=30e6):
    '''
    Expects individual row vectors.
    All numpy array shapes expected to match.
    '''
    
    lightspd = 299792458.0
    
    radial = t_x - r_x # convention pointing towards transmitter
    radial_n = radial / np.linalg.norm(radial,axis=1).reshape((-1,1)) # don't remove this reshape, nor the axis arg
    
    if radial_n.ndim == 1:
        vradial = np.dot(radial_n, r_xdot) - np.dot(radial_n, t_xdot) # minus or plus?
    else:
        vradial = np.zeros(len(radial_n))
        for i in range(len(radial_n)):
            vradial[i] = np.dot(radial_n[i,:],r_xdot[i,:]) - np.dot(radial_n[i,:], t_xdot[i,:])
    
    foa = vradial/lightspd * freq
    
    return foa
    