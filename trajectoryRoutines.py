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
        DESCRIPTION. The default is None.
        The output will use this as the coefficient along the connecting vector between
        the 2 anchor points, as the position to start iterating at.
    randomWalk : Scalar, optional
        DESCRIPTION. The default is None.
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