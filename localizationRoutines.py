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

#%% Hyperbola routines
def hyperbolaGradDesc(pt, s1, s2, rangediff, step, epsilon):
    # Define the gradient function for the hyperbola cost
    grad = lambda x: ((x-s2)/np.linalg.norm(s2) - (x-s1)/np.linalg.norm(s1)) * (np.linalg.norm(s2-x) - np.linalg.norm(s1-x) - rangediff) / np.abs((np.linalg.norm(s2-x) - np.linalg.norm(s1-x) - rangediff))
    
    history = [pt]
    initgrad = np.zeros(3)
    while np.abs((np.linalg.norm(s2-pt) - np.linalg.norm(s1-pt) - rangediff)) != 0 and np.linalg.norm(grad(pt)) * step > epsilon:
        newgrad = grad(pt)
        if np.dot(initgrad, newgrad) < 0:
            print('gradient reversed')
            # Lower stepsize a bit
            step = step / 2 # TODO: to optimise
        pt = pt - step * newgrad
        history.append(pt)
        initgrad = newgrad

    return history

def generateHyperbolaFlat(
        rangediff: float,
        s1: np.ndarray, s2: np.ndarray, z: float=0, startpt: np.ndarray=None,
        initstep: float=0.1, epsilon: float=1e-8, orthostep:float = 0.1):
    
    if startpt is None:
        # Generate a start point by the mid-point of the two sensors
        startpt = (s1+s2) / 2.0
        startpt[2] = z # Fix the z-value
       
    # Begin the gradient descent for the start point
    startHistory = hyperbolaGradDesc(startpt, s1, s2, rangediff, initstep, epsilon)
    
    
    
    
    
    
    


#%%
def gridSearchBlindLinearRTT(
        tx_list: np.ndarray,
        rx_list: np.ndarray,
        time_list: np.ndarray,
        toa_list: np.ndarray,
        toa_sigma_list: np.ndarray,
        grid_list: np.ndarray,
        verb: bool=True):
    # Docstring here..
    
    # Instantiate output
    numGridPts = grid_list.shape[0]
    cost_grid = np.zeros(numGridPts)
    lightspd = 299792458.0
    
    # Change dimensions if necessary
    N = toa_list.size # numMeasurements
    if rx_list.ndim == 1:
        rx_list = np.tile(rx_list, N).reshape((N,-1))
    if tx_list.ndim == 1:
        tx_list = np.tile(tx_list, N).reshape((N,-1))
        
    # Convert time into matrix for least squares later
    A = np.hstack((
        time_list.reshape((-1,1)),
        np.ones((time_list.size,1))
    ))
    
    # breakpoint()
    
    
    for gi, gridpt in enumerate(grid_list):
        # Define theoretical time segments for all measurements
        time_x2rx = np.linalg.norm(rx_list - gridpt, axis=1) / lightspd
        time_tx2x = np.linalg.norm(tx_list - gridpt, axis=1) / lightspd
        # Column vector, defines the theoretical total TOA based on distance alone
        gamma = (time_x2rx + time_tx2x).reshape((-1,1))
        # Define extra delay (d) as the difference observed
        d = toa_list.reshape((-1,1)) - gamma
        # Fit least squares
        soln, residuals, rank, singulars = np.linalg.lstsq(A, d)
        
        # Save residuals
        cost_grid[gi] = np.sum(residuals)
        
    return cost_grid
    
    
    


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
        
        # Calculate direction vectors from sensors to the grid
        dirvecm1 = fullmesh - s1x
        dirvecm2 = fullmesh - s2x
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

def gridSearchTDFD_direct(s1x_list, s2x_list,
                          tdoa_list, td_sigma_list,
                          s1v_list, s2v_list,
                          fdoa_list, fd_sigma_list, fc,
                          gridmat, verb=True):
    
    lightspd = 299792458.0
    cost_grid = np.zeros(gridmat.shape[0])
    
    # Pre-normalize fdoa by the fc
    nfdoa_list = fdoa_list / fc
    nfd_sigma_list = fd_sigma_list / fc
    
    # Pre-scale by lightspd
    r_list = (tdoa_list * lightspd).astype(np.float32)
    r_sigma_list = (td_sigma_list * lightspd).astype(np.float32)
    drdt_list = (nfdoa_list * lightspd).astype(np.float32)
    drdt_sigma_list = (nfd_sigma_list * lightspd).astype(np.float32)
    
    t1g = time.time()
    for i in range(len(tdoa_list)):
        # cpu code
        s1x = s1x_list[i]
        s2x = s2x_list[i]
        # tdoa = tdoa_list[i]
        r = r_list[i]
        # td_sigma = td_sigma_list[i]
        r_sigma = r_sigma_list[i]
        
        s1v = s1v_list[i]
        s2v = s2v_list[i]
        # fdoa = nfdoa_list[i]
        drdt = drdt_list[i]
        # fd_sigma = nfd_sigma_list[i]
        drdt_sigma = drdt_sigma_list[i]
        
        # TD related
        # r = np.float32(tdoa * lightspd)
        # r_sigma = np.float32(td_sigma * lightspd)
        
        rm = np.linalg.norm(s2x - gridmat, axis=1) - np.linalg.norm(s1x - gridmat, axis=1)
        # TD cost
        td_cost = ((r - rm) / r_sigma)**2
        
        # FD related
        # drdt = fdoa * lightspd
        # drdt_sigma = fd_sigma * lightspd
        
        # Calculate direction vectors from sensors to the grid
        dirvecm1 = gridmat - s1x
        dirvecm2 = gridmat - s2x
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
        # FD cost
        fd_cost = ((drdt - vmdiff) / drdt_sigma)**2
        # print(np.min(fd_cost))
        
        # Accumulate costs
        np.add(cost_grid, td_cost, out=cost_grid)
        np.add(cost_grid, fd_cost, out=cost_grid)
        
    
    t2g = time.time()
    if verb:
        print("Grid search took %g seconds." % (t2g-t1g))
    
    return cost_grid

#%% CRB Routines (conversions from commonMex)
def calcCRB_TD(x, S, sig_r, pairs=None, cmat=None):
    ''' S is presented column-wise i.e. 3 X N array. '''
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

def calcCRB_TDFD(x, S, sig_r, xdot, Sdot, sig_r_dot, pairs=None, cmat=None):
    ''' S is presented column-wise i.e. 3 X N array. '''
    if x.ndim == 1:
        x = x.reshape((-1,1)) # Reshapes do not alter the external array (the one passed in)
    if xdot.ndim == 1:
        xdot = xdot.reshape((-1,1))
        
    m = S.shape[1] # no. of sensors
    r = np.linalg.norm(x - S, axis=0)
    r_dx = (x - S) / r
    
    rdot = np.sum((xdot - Sdot) * (x - S), axis=0) / r
    r_dxdot = np.zeros((3,m))
    rdot_dx = (-r_dx * rdot + xdot - Sdot) / r
    rdot_dxdot = (x - S) / r
    
    if pairs is None: # Assume every pair in S is used with no overlaps
        pairs = np.arange(m).reshape((-1,2))
    
    numPairs = pairs.shape[0]
    R = np.zeros((6,numPairs))
    Rdot = np.zeros((6,numPairs))
    
    for k in np.arange(numPairs):
        c1 = pairs[k,0]
        c2 = pairs[k,1]
        
        R[0:3, k] = r_dx[:, c1] - r_dx[:, c2]
        R[3:6, k] = r_dxdot[:, c1] - r_dxdot[:, c2]
        Rdot[0:3, k] = rdot_dx[:, c1] - rdot_dx[:, c2]
        Rdot[3:6, k] = rdot_dxdot[:, c1] - rdot_dxdot[:,c2]
        
    SIGR = np.diag(sig_r**-2)
    SIGRDOT = np.diag(sig_r_dot**-2)
    FIM_R = R @ SIGR @ R.T
    FIM_Rdot = Rdot @ SIGRDOT @ Rdot.T
    FIM = FIM_R + FIM_Rdot
    
    if cmat is None:
        crb = np.linalg.inv(FIM)
    else:
        U = sp.linalg.null_space(cmat.T)
        crb = U @ np.linalg.inv(U.T @ FIM @ U) @ U.T
        
    return crb

def calcCRB_BlindLinearRTT(x, S, P, t, sig_r, cmat=None):
    ''' S is presented column-wise i.e. 3 X N array. '''
    if x.ndim == 1:
        x = x.reshape((-1,1)) # Reshapes do not alter the external array (the one passed in)
    if P.ndim == 1:
        P = P.reshape((-1,1))
    
    m = S.shape[1] # no. of sensors
    rS = np.linalg.norm(x - S, axis=0)
    rP = np.linalg.norm(x - P, axis=0)
    r_dx = (x - S) / rS + (x - P) / rP
    
    # No need to define r_db, since all ones
    # No need to define r_da, since it is just the t vector
    
    R = np.zeros((5, m))
    R[0:3] = r_dx
    R[3] = t
    R[4] = 1
        
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

