# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 13:47:58 2021

@author: Lken
"""

import numpy as np
import scipy as sp
from scipy.stats.distributions import chi2
from numba import jit
import time

#%%
# @jit(nopython=True) # not working until numba includes axis option in linalg.norm
def gridSearchRTT(
        t_list: np.ndarray,
        r_list: np.ndarray,
        toa_list: np.ndarray,
        toa_sigma_list: np.ndarray,
        grid_list: np.ndarray,
        verb: bool=True):
    '''
    Localizes based on a one-bounce RTT measurement.
    
    Parameters
    ----------
    t_list : np.ndarray
        Transmitter position. If 1-d array, assumed as a static position,
        otherwise each row represents a position for the associated measurement.
    r_list : np.ndarray
        Receiver position (may be the same as the transmitter). 
        If 1-d array, assumed as a static position,
        otherwise each row represents a position for the associated measurement.
    toa_list : np.ndarray
        RTT time-of-arrival measurements.
    toa_sigma_list : np.ndarray
        RTT measurement errors.
    grid_list : np.ndarray
        2-d array of grid points to evaluate costs at. Each row represents a point.
    verb : bool, optional
        Verbose printing (for timing). The default is True.

    Returns
    -------
    cost_grid : np.ndarray
        A cost array of length equal to the number of grid points.

    '''
    
    # Instantiate output
    numGridPts = grid_list.shape[0] # Each row is a point
    cost_grid = np.zeros(numGridPts)
    t1g = time.time()
    
    for i in range(len(toa_list)):
        if t_list.ndim == 1: # Then static tx
            t = t_list
        else:
            t = t_list[i,:]
            
        if r_list.ndim == 1: # Then static rx
            r = r_list
        else:
            r = r_list[i,:]
            
        # Compute expectation for the grid points
        e_dist = np.linalg.norm(t - grid_list, axis=1) + np.linalg.norm(r - grid_list, axis=1)
        
        # Compute distance, error from TOA
        m_dist = 299792458.0 * toa_list[i]
        m_err = 299792458.0 * toa_sigma_list[i]
        
        # Add to the grid
        cost_grid = cost_grid + (e_dist - m_dist)**2 / m_err**2
    
    t2g = time.time()
    if verb:
        print("Grid search took %g seconds." % (t2g-t1g))
        
    return cost_grid
    


def gridSearchTDOA(s1x_list, s2x_list, tdoa_list, td_sigma_list, xrange, yrange, z, verb=True):
    '''
    Assumes a flat surface.
    z: height of surface
    '''
    xm, ym = np.meshgrid(xrange,yrange)
    fullmesh = np.vstack((xm.flatten(),ym.flatten(),np.zeros(len(ym.flatten())) + z)).transpose().astype(np.float32)
    cost_grid = 0
    
    t1g = time.time()
    for i in range(len(tdoa_list)):
        # cpu code
        s1x = s1x_list[i].astype(np.float32)
        s2x = s2x_list[i].astype(np.float32)
        tdoa = tdoa_list[i].astype(np.float32)
        td_sigma = td_sigma_list[i].astype(np.float32)
        
        r = np.float32(tdoa * 299792458.0)
        r_sigma = np.float32(td_sigma * 299792458.0)
        
        rm = np.linalg.norm(s2x - fullmesh, axis=1) - np.linalg.norm(s1x - fullmesh, axis=1)
        
        if cost_grid is None:
            cost_grid = (r - rm)**2 / r_sigma**2
        else:
            cost_grid = cost_grid + (r - rm)**2 / r_sigma**2
        
    t2g = time.time()
    if verb:
        print("Grid search took %g seconds." % (t2g-t1g))
    
    return cost_grid

def gridSearchFDOA(s1x_list, s2x_list, s1v_list, s2v_list, fdoa_list, fd_sigma_list, xrange, yrange, z, fc, verb=True):
    xm, ym = np.meshgrid(xrange,yrange)
    fullmesh = np.vstack((xm.flatten(),ym.flatten(),np.zeros(len(ym.flatten())) + z)).transpose().astype(np.float32)
    cost_grid = 0
    
    # Pre-normalize fdoa by the fc
    nfdoa_list = fdoa_list / fc
    nfd_sigma_list = fd_sigma_list / fc
    
    for i, fdoa in enumerate(nfdoa_list):
        s1x = s1x_list[i].astype(np.float32)
        s2x = s2x_list[i].astype(np.float32)
        s1v = s1v_list[i].astype(np.float32)
        s2v = s2v_list[i].astype(np.float32)
    
        fd_sigma = nfd_sigma_list[i].astype(np.float32)
        
        # Range rate
        drdt = np.float32(fdoa * 299792458.0)
        drdt_sigma = np.float32(fd_sigma * 299792458.0)
        
        # Do we need this?
        rm = np.linalg.norm(s2x - fullmesh, axis=1) - np.linalg.norm(s1x - fullmesh, axis=1)
        
        # Calculate direction vectors from the grid
        dirvecm1 = s1x - fullmesh
        dirvecm2 = s2x - fullmesh
        # Need the normalized versions
        dirvecm1 = dirvecm1 / np.linalg.norm(dirvecm1, axis=1).reshape((-1,1))
        dirvecm2 = dirvecm2 / np.linalg.norm(dirvecm2, axis=1).reshape((-1,1))
        # We want the component of velocity along the direction vectors
        parvm1 = np.dot(dirvecm1, s1v)
        parvm2 = np.dot(dirvecm2, s2v) # This should already be negative when direction and velocities are opposed
        # print(parvm1)
        # print(parvm2)
        # For each velocity calculated, compute the range rate difference
        # as the metric
        vmdiff = parvm2 - parvm1
        # print(vmdiff)
        
        if cost_grid is None:
            cost_grid = (drdt - vmdiff)**2 / drdt_sigma**2
        else:
            cost_grid = cost_grid + (drdt - vmdiff)**2 / drdt_sigma**2
            
    return cost_grid
        
        

def gridSearchTDOA_direct(s1x_list, s2x_list, tdoa_list, td_sigma_list, gridmat, verb=True):
    '''
    
    Parameters
    ----------
    gridmat : np.ndarray
        N x 3 array, where N is the number of grid points in total; each row is the x,y,z values.


    '''
    
    cost_grid = None
    t1g = time.time()
    for i in range(len(tdoa_list)):
        # cpu code
        s1x = s1x_list[i]
        s2x = s2x_list[i]
        tdoa = tdoa_list[i]
        td_sigma = td_sigma_list[i]
        
        r = np.float32(tdoa * 299792458.0)
        r_sigma = np.float32(td_sigma * 299792458.0)
        
        rm = np.linalg.norm(s2x - gridmat, axis=1) - np.linalg.norm(s1x - gridmat, axis=1)
        
        if cost_grid is None:
            cost_grid = (r - rm)**2 / r_sigma**2
        else:
            cost_grid = cost_grid + (r - rm)**2 / r_sigma**2
    
    t2g = time.time()
    if verb:
        print("Grid search took %g seconds." % (t2g-t1g))
    
    return cost_grid

#%% CRB Routines
def calcCRB_TD(x, S, sig_r, pairs=None, cmat=None):
    if x.ndim == 1:
        x = x.reshape((-1,1)) # Reshapes do not alter the external array (the one passed in)
    
    m = S.shape[1] # no. of sensors
    r = np.linalg.norm(x - S, axis=0)
    r_dx = (x - S) / r
    
    if pairs is None: # Assume every pair in S is used with no overlaps
        pairs = np.arange(m).reshape((-1,2))
        
    numPairs = pairs.shape[0]
    R = np.zeros((3,numPairs))
    
    for k in np.arange(numPairs):
        R[:3, k] = r_dx[:, pairs[k,0]] - r_dx[:, pairs[k,1]]
        
    SIGR = np.diag(sig_r**-2)
    FIM = R @ SIGR @ R.T
    
    if cmat is None:
        crb = np.linalg.inv(FIM)
    else:
        U = sp.linalg.null_space(cmat.T)
        crb = U @ np.linalg.inv(U.T @ FIM @ U) @ U.T
        
    return crb

def projectCRBtoEllipse(crb, pos, percent, dof=2, theta=None):
    if pos.ndim == 1:
        pos = pos.reshape((-1,1))
    
    sigval = chi2.ppf(percent, df=dof)
    u, s, vh = np.linalg.svd(crb)
    a = s[0]**0.5
    b = s[1]**0.5
    
    if theta is None:
        theta = np.arange(0,2*np.pi,0.01)
        
    r = sigval**0.5 * a * b / np.sqrt(b**2 * np.cos(theta)**2  + a**2 * np.sin(theta)**2)
    
    x = np.repeat(np.expand_dims(r * np.cos(theta), 0), 3, axis=0)
    y = np.repeat(np.expand_dims(r * np.sin(theta), 0), 3, axis=0)
    ellipse = x * u[:,0].reshape((-1,1)) + y * u[:,1].reshape((-1,1)) + pos
    
    return ellipse
    
    

#%%
try:
    import cupy as cp
    
    gridsearchtdoa_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void gridsearchtdoa_kernel(int len, float *s1x_list, float *s2x_list,
                               float *tdoa_list, float *td_sigma_list,
                               float x0, float y0, float xp, float yp,
                               int xn, int yn, float z,
                               float *cost_grid)
    {
        // allocate shared memory
        extern __shared__ float s[];
        float *s_s1x_l = s; // (len * 3) floats
        float *s_s2x_l = (float*)&s_s1x_l[len * 3]; // (len * 3) floats
        float *s_r_l = (float*)&s_s2x_l[len * 3]; // (len) floats
        float *s_rsigma_l = (float*)&s_r_l[len]; // (len) floats
        
        // load shared memory
        for (int t = threadIdx.x; t < len * 3; t = t + blockDim.x){
            s_s1x_l[t] = s1x_list[t];
            s_s2x_l[t] = s2x_list[t];
        }
        for (int t = threadIdx.x; t < len; t = t + blockDim.x){
            s_r_l[t] = tdoa_list[t] * 299792458.0;
            s_rsigma_l[t] = td_sigma_list[t] * 299792458.0; // perform the multiplies while loading
        }
        
        __syncthreads();
        
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        int row = tid / xn;
        int col = tid % xn;
        
        float x = col * xp + x0;
        float y = row * yp + y0;
        
        float rm, rm2, rm1;
        float fullcost = 0.0f;
        float cost;
        
        for (int i = 0; i < len; i++){
            
            rm1 = norm3df(s_s1x_l[i*3+0] - x, s_s1x_l[i*3+1] - y, s_s1x_l[i*3+2] - z);
            rm2 = norm3df(s_s2x_l[i*3+0] - x, s_s2x_l[i*3+1] - y, s_s2x_l[i*3+2] - z);
            rm = rm2 - rm1; // theoretical range for the point
            
            cost = (s_r_l[i] - rm) / s_rsigma_l[i];
     
            fullcost = fmaf(cost, cost, fullcost); // accumulate costs (len) times
            
        }
        
        // write to output
        int cost_grid_len = xn * yn;
        if (tid < cost_grid_len){
            cost_grid[tid] = fullcost;
        }
    }
    ''', 'gridsearchtdoa_kernel')

    def gridSearchTDOA_gpu(s1x_list, s2x_list, tdoa_list, td_sigma_list, xrange, yrange, z, verb=True, moveToCPU=False):
        d_s1x_l = cp.asarray(s1x_list).astype(cp.float32)
        d_s2x_l = cp.asarray(s2x_list).astype(cp.float32)
        d_tdoa_l = cp.asarray(tdoa_list).astype(cp.float32)
        d_tdsigma_l = cp.asarray(td_sigma_list).astype(cp.float32)
        
        x0 = np.min(xrange).astype(np.float32)
        xp = (xrange[1]-xrange[0]).astype(np.float32)
        xn = len(xrange)
        y0 = np.min(yrange).astype(np.float32)
        yp = (yrange[1]-yrange[0]).astype(np.float32)
        yn = len(yrange)
        
        # prepare output
        d_cost_grid = cp.zeros(xn*yn, dtype=cp.float32)
        
        # run kernel
        t1 = time.time()
        THREADS_PER_BLOCK = 128
        NUM_BLOCKS = int(d_cost_grid.size/THREADS_PER_BLOCK + 1)
        gridsearchtdoa_kernel((NUM_BLOCKS,),(THREADS_PER_BLOCK,), (len(s1x_list), d_s1x_l, d_s2x_l, d_tdoa_l, d_tdsigma_l,
                                                                   x0, y0, xp, yp, xn, yn, z, d_cost_grid), 
                              shared_mem=(d_s1x_l.size + d_s2x_l.size + d_tdoa_l.size + d_tdsigma_l.size) * 4)
        t2 = time.time()
        
        if verb:
            print("Grid search kernel took %g seconds." % (t2-t1))
        
        if moveToCPU:
            return cp.asnumpy(d_cost_grid)
        else:
            return d_cost_grid
        
        

        
            
except:
    print("Cupy unavailable. GPU routines not imported.")

