# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 13:47:58 2021

@author: Lken
"""

import numpy as np
import scipy as sp
from scipy.stats.distributions import chi2
# from numba import jit, njit
import time
from skyfield.api import wgs84, load, Distance
from skyfield.toposlib import ITRSPosition

from plotRoutines import *
from satelliteRoutines import *
import skyfield.api

from enum import Enum

#%% WGS84 constants
class WGS84Coefficients(Enum):
    a = 6378137.0
    b = 6356752.314245 # WGS84 constants, reference https://en.wikipedia.org/wiki/World_Geodetic_System


#%% Coordinate transformations
def geodeticLLA2ecef(lat_rad, lon_rad, h, checkRanges=False):
    # Some error checking
    if checkRanges and (np.any(np.abs(lat_rad) > np.pi/2) or np.any(np.abs(lon_rad) > np.pi)):
        raise ValueError("Latitude and longitude magnitudes are too large. Did you forget to convert to radians?")
    # This should replicate wgs84.latlon().itrs_xyz.m (which also works on arrays)
    # Speedwise, this is about 2-3x faster, since it skips all the object instantiations
    # Reference https://en.wikipedia.org/wiki/Geodetic_coordinates
    a = 6378137.0
    b = 6356752.314245 # WGS84 constants, reference https://en.wikipedia.org/wiki/World_Geodetic_System
    N = a**2 / np.sqrt(a**2 * np.cos(lat_rad)**2 + b**2 * np.sin(lat_rad)**2)
    
    x = (N+h)*np.cos(lat_rad)*np.cos(lon_rad)
    y = (N+h)*np.cos(lat_rad)*np.sin(lon_rad)
    z = (b**2/a**2 * N + h) * np.sin(lat_rad)
    
    return np.vstack((x,y,z))

def ecef2geodeticLLA(x: np.ndarray):
    if not isinstance(x, np.ndarray):
        raise TypeError("Must be numpy array.")
    
    now = load.timescale().now() # Can re-use since ITRS is time-agnostic

    # A single 3-d position
    if x.ndim == 1 and x.size == 3:
        x = x.reshape((1, 3))
        
    # Multiple 3-d positions, one in each column
    if x.shape[0] == 3:
        pos = ITRSPosition(Distance(m=x)) # This accepts a 3xN array directly
        latlonele = wgs84.geographic_position_of(pos.at(now))
        lle_arr = np.vstack(
            (latlonele.latitude.degrees, latlonele.longitude.degrees, latlonele.elevation.m)
        )
        return lle_arr

    else:
        raise ValueError("Invalid dimensions; expected 3xN.")

#%% Doppler convention routines
def calculateRangeRate(
    tx_x: np.ndarray,
    rx_x: np.ndarray,
    tx_xdot: np.ndarray=np.zeros(3),
    rx_xdot: np.ndarray=np.zeros(3)
):
    # We use the direction of the wave
    dirvec = rx_x - tx_x
    # Normalise by it
    if dirvec.ndim == 1:
        dirvec = dirvec.reshape((1,3)) # Ensure its a 2D row
    elif dirvec.shape[1] != 3:
        raise ValueError("Direction vectors (rx_x - tx_x) must be a Nx3 array.")
    dirvec = dirvec/(np.linalg.norm(dirvec, axis=1).reshape((-1,1)))
    # Find the speed parallel to this direction vector
    tx_xdot_p = np.dot(dirvec, tx_xdot) # This ordering of dirvec first lets us have multiple rows of dirvecs dotted with a single 1D xdot array
    rx_xdot_p = np.dot(dirvec, rx_xdot)
    
    # The range rate is characterised by the (rx speed - tx speed)
    # This can be seen by noting the conditions for rx and tx respectively
    # If RX moves away from the incoming wave ie same direction, 
    # range rate is increasing, and vice versa.
    # If TX moves back from the outgoing wave ie opposite direction,
    # range rate is increasing, and vice versa.
    rdot = rx_xdot_p - tx_xdot_p

    return rdot

def calculateDoppler(
    f0: float,
    tx_x: np.ndarray,
    rx_x: np.ndarray,
    tx_xdot: np.ndarray=np.zeros(3),
    rx_xdot: np.ndarray=np.zeros(3),
    lightspd: float=299792458.0
):
    # To calculate Doppler, simply get the range rate
    rdot = calculateRangeRate(tx_x, rx_x, tx_xdot, rx_xdot)
    # And then take the reverse of it with a scaling
    doppler = -rdot/lightspd*f0
    return doppler



#%% Hyperbola routines
# @njit(nogil=True)
def rangeOfArrival(x, s_i):
    """
    Computes the range of arrival between vectors.
    This is simply the norm of the difference between the vectors.
    The shapes of the two inputs are irrelevant as long as they are broadcastable via numpy's rules,
    but the norm is always taken over the last index.

    For the common case of 2-D arrays, this means that each row of the input array is treated as a vector.
    E.g. for M x 3 arrays, the range of arrivals is a length M vector.

    Parameters
    ----------
    x : array_like
        The first vector.
    s_i : array_like
        The second vector.
        Naming convention is simply due to the 'sensor - target x' convention.

    Returns
    -------
    rho : array_like
        The range of arrival.
    """
    rho = np.linalg.norm(x-s_i, axis=-1)
    return rho

# @njit(nogil=True)
def rangeDifferenceOfArrival(x, s1, s2):
    return rangeOfArrival(x,s2) - rangeOfArrival(x,s1)

# @njit(nogil=True)
def rangeOfArrivalGradient(x, s_i):
    rho = rangeOfArrival(x, s_i)
    return (x - s_i) / rho

def hyperboloidLineIntersectCostFunc(delta, x0, s1, s2, rangediff, g):
    return (rangeOfArrival(x0+g*delta, s2) - rangeOfArrival(x0+g*delta, s1) - rangediff)**2
   
# @njit(nogil=True) 
def hyperboloidGradient(x, s1, s2, rangediff):
    rho1 = rangeOfArrival(x, s1)
    rho2 = rangeOfArrival(x, s2)
    g = 2*(rho2 - rho1 - rangediff) * (rangeOfArrivalGradient(x, s2) - rangeOfArrivalGradient(x, s1))
    return g

# @njit(nogil=True)
def hyperbolaGradDesc(pt, s1, s2, rangediff, step, epsilon, surfaceNorm=np.array([0,0,1],dtype=np.float64), verb=False):
    # Note that default surface normal vector is parallel to axis,
    # ie the default is planes parallel to XY.
    
    # Ensure surfaceNorm is unit vector
    surfaceNorm = surfaceNorm / np.linalg.norm(surfaceNorm)
    
    # Use scipy.optimize to minimize, seems like a 33% reduction in calculation time compared to the old code
    g = hyperboloidGradient(pt, s1, s2, rangediff) # Calculate gradient at the point, use as a line
    g = g - np.dot(surfaceNorm, g)*surfaceNorm # Project onto surface
    
    
    # ### MANUAL LOOP
    # cnt = 0
    # while (np.linalg.norm(step * g) > epsilon) and (rangeDifferenceOfArrival(pt,s1,s2) != rangediff):
    #     pt = pt - step*g
    #     gnew = hyperboloidGradient(pt, s1, s2, rangediff) # Calculate gradient at the point, use as a line
    #     gnew = gnew - np.dot(surfaceNorm, gnew)*surfaceNorm # Project onto surface
    #     if np.dot(gnew, g) < 0:
    #         # print('reversal')
    #         step = step * 0.25
    #     g = gnew
    #     # print(pt)
    #     # breakpoint()
    #     cnt += 1
    
    # # print(cnt)
    # return pt
    
    ### SCIPY.OPTIMIZE VERSION
    g = g/np.linalg.norm(g)
    result = sp.optimize.minimize(hyperboloidLineIntersectCostFunc, 0, args=(pt, s1, s2, rangediff, g))
    
    val = result.x*g+pt
    # if result.x[0] == 0:
    #     print(result.nit)
    #     print("EH?")
    #     print(np.linalg.norm(g))

    # breakpoint()

    return val
    
# @njit(nogil=True)
def hyperbolaTangentXY(pt, s1, s2, rangediff):
    hz = 0 # This is constant for our flat, non-angled plane, regardless of z-value
    # Vector satisfies g . h = 0, so gx * hx + gy * hy = 0
    # Hence hy = -gx/gy hx
    hx = 1.0
    g = hyperboloidGradient(pt, s1, s2, rangediff)
    if g[1] == 0.0:
        hy = 1.0
        hx = 0.0
    else:
        hy = -g[0] / g[1]
    
    h = np.array([hx, hy, hz])
    h = h / np.linalg.norm(h) # Normalise
    
    return h

# @njit(nogil=True)
def generateHyperbolaXY(
        halfNumPts: int, rangediff: float,
        s1: np.ndarray, s2: np.ndarray, z: float=0, startpt: np.ndarray=None,
        initstep: float=0.1, epsilon: float=1e-8, orthostep:float = 0.1):
    
    if startpt is None:
        # Generate a start point by the mid-point of the two sensors
        startpt = (s1+s2) / 2.0
        startpt[2] = z # Fix the z-value
       
    # Begin the gradient descent for the start point
    startpt = hyperbolaGradDesc(startpt, s1, s2, rangediff, initstep, epsilon)
    
    
    # Find the two tangent vectors to it, in the plane
    h_1 = hyperbolaTangentXY(startpt, s1, s2, rangediff)
    h_2 = -h_1
    
    # Propagate for the number of points
    h = h_1 # Initial tangent vector
    pt = startpt # Initial point
    
    # Temporary vectors
    oldpt = np.zeros(3, dtype=np.float64)
    hnew = np.zeros(3, dtype=np.float64)
    
    # Output vector
    hyperbola = np.zeros((2*halfNumPts+1, 3),dtype=np.float64)
    hyperbola[halfNumPts,:] = startpt[:] # Initial point
    
    # pts_1 = np.zeros((halfNumPts, 3))
    for i in np.arange(halfNumPts):
        oldpt[:] = pt[:]
        # First move by the tangent vector
        pt = pt + h * orthostep
        
        # Then descent back to the hyperbola
        pt = hyperbolaGradDesc(pt, s1, s2, rangediff, initstep, epsilon)
        # print(rangeOfArrival(pt, s2)-rangeOfArrival(pt,s1))
        
        # Accumulate the point
        hyperbola[halfNumPts-i-1,:] = pt
        # pts_1[-i-1] = pt
        
        # Get the new tangent vector
        np.subtract(pt, oldpt, out=hnew)
        # hnew = pt - oldpt # We can move by just extension instead of calculating tangent
        h = hnew/np.linalg.norm(hnew)
        # print(h)
        
        # hnew = hyperbolaTangentXY(pt, s1, s2, rangediff)
        # h = hnew * np.sign(np.dot(hnew,h))
            
    # End of loop
        
    # Propagate for the number of points on other side as well
    h = h_2 # Initial tangent vector
    pt = startpt # Initial point
    # pts_2 = np.zeros((halfNumPts, 3))
    for i in np.arange(halfNumPts):
        oldpt[:] = pt[:]
        # First move by the tangent vector
        pt = pt + h * orthostep
        
        # Then descent back to the hyperbola
        pt = hyperbolaGradDesc(pt, s1, s2, rangediff, initstep, epsilon)
        
        # Accumulate the point
        hyperbola[halfNumPts+i+1,:] = pt
        # pts_2[i] = pt
        
        # Get the new tangent vector
        np.subtract(pt, oldpt, out=hnew)
        # hnew = pt - oldpt # We can move by just extension instead of calculating tangent
        h = hnew/np.linalg.norm(hnew)
        
        
        # hnew = hyperbolaTangentXY(pt, s1, s2, rangediff)
        # h = hnew * np.sign(np.dot(hnew,h))
        
    # End of loop
    
    # Attach all the points together (already sorted)
    # hyperbola = np.vstack((pts_1, startpt, pts_2))
    
    return hyperbola
    
    
    
    
    
    
    


#%%
def gridSearchBlindLinearRTT(
        tx_list: np.ndarray,
        rx_list: np.ndarray,
        time_list: np.ndarray,
        toa_list: np.ndarray,
        toa_sigma_list: np.ndarray,
        grid_list: np.ndarray,
        verb: bool=True):
    
    '''
    Parameters
    ----------
    tx_list : np.ndarray
        Transmitter position(s). If 1-d array, assumed as a static position,
        otherwise each row represents a position for the associated measurement.
    rx_list : np.ndarray
        Receiver position (may be the same as the transmitter). 
        If 1-d array, assumed as a static position,
        otherwise each row represents a position for the associated measurement.
    time_list : np.ndarray
        Time value to perform linear scaling on. Often you can use the transmit time or receive time.
    toa_list : np.ndarray
        RTT time-of-arrival measurements. These should be strictly positive.
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

#%% Grid generator for most localizers
def latlongrid_to_ecef(centrelat, centrelon, latspan, lonspan, numLat, numLon):
    lonlist = np.linspace(centrelon - lonspan/2, centrelon + lonspan/2, numLon)
    latlist = np.linspace(centrelat - latspan/2 ,centrelat + latspan/2, numLat)
    longrid, latgrid = np.meshgrid(lonlist, latlist)
    
    longridflat = longrid.flatten()
    latgridflat = latgrid.flatten()
    
    # Convert to xyz
    ecefgrid = wgs84.latlon(latgridflat, longridflat).itrs_xyz.m.transpose() # N x 3
    
    return ecefgrid, lonlist, latlist
    
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
        
    return crb, FIM

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


#%% Helper classes
class GridLocalizer:
    lightspd = 299792458.0
    
    def __init__(self, gridmat: np.ndarray, xrange: np.ndarray, yrange: np.ndarray):
        '''
        Initialises a localizer which will search a grid of points.

        Parameters
        ----------
        gridmat : np.ndarray
            N x 3 matrix of N Cartesian coordinate points i.e. each point is 1 row.
            Does not technically need to be evenly spaced.
            See classmethods for common ways to instantiate.
        xrange : np.ndarray
            Length L array of x values used to generate the gridmat.
        yrange : np.ndarray
            Length M array of y values used to generate the gridmat. 
        '''
        self.gridmat = gridmat
        self.xrange = xrange
        self.yrange = yrange
       
    # Factory methods
    @classmethod
    def fromXYMeshgrid(cls, xrange: np.ndarray, yrange: np.ndarray):
        xm, ym = np.meshgrid(xrange, yrange)
        gridmat = np.hstack((xm.reshape((-1,1)), ym.reshape((-1,1))))
        return cls(gridmat, xrange, yrange)
        
    # Interface methods (see mixins below)
    def run(self):
        raise NotImplementedError("This method is only defined in subclasses.")
        
    def localize(self, cost_grid: np.ndarray):
        gridminidx = np.argmin(cost_grid)
        gridmin = self.gridmat[gridminidx]
        return gridmin
        
    def crb(self):
        raise NotImplementedError("This method is only defined in subclasses.")
        
    def plot(self, cost_grid: np.ndarray):
        '''
        Plots a heatmap of the cost_grid generated from run().
        You should not need to reshape the output from run() yourself.

        Parameters
        ----------
        cost_grid : np.ndarray
            Length N array of least squares errors for each grid point, 
            directly from the run() method. All reshaping is done automatically, do not reshape it yourself!

        Returns
        -------
        ax : pyqtgraph axes
            The Pyqtgraph axes. You can call ax.plot() again to plot on top of the heatmap.
        img : pyqtgraph.ImageItem
            The Pyqtgraph heatmap image.

        '''
        ax, img = pgPlotHeatmap(np.exp(-0.5*cost_grid.reshape((self.yrange.size, self.xrange.size))).T, # must transpose
                                self.xrange[0],
                                self.yrange[0],
                                np.ptp(self.xrange),
                                np.ptp(self.yrange),
                                autoBorder=True)
        return ax, img
    
#%%
class LatLonGridLocalizer(GridLocalizer):
    def __init__(self, latlist: np.ndarray, lonlist: np.ndarray, gridmat: np.ndarray):
        '''
        Initialises a localizer based on a latitude-longitude grid rather than
        Cartesian coordinates. Calculations will still be performed in Cartesian space.

        Parameters
        ----------
        latlist : np.ndarray
            The array of latitude values.
        lonlist : np.ndarray
            The array of longitude values.
        gridmat : np.ndarray
            The associated matrix of Cartesian points. See GridLocalizer's __init__ method.
            See factory methods like 'fromLatLonLimits' for easy initializations.

        '''
        super().__init__(gridmat, lonlist, latlist)
        # We repoint some variable names for clarity
        self.lonlist = lonlist
        self.latlist = latlist
        
    @classmethod
    def fromLatLonLimits(cls, centrelat, centrelon, latspan, lonspan, numLat, numLon):
        ecefgrid, lonlist, latlist = latlongrid_to_ecef(centrelat, centrelon, latspan, lonspan, numLat, numLon)
        return cls(latlist, lonlist, ecefgrid)
    
    def localize(self, cost_grid: np.ndarray):
        gridminidx = np.argmin(cost_grid)
        latgridmin = gridminidx // self.latlist.size
        longridmin = gridminidx % self.latlist.size
        longridmin = self.lonlist[longridmin]
        latgridmin = self.latlist[latgridmin]
        gridmin = self.gridmat[gridminidx]
        return longridmin, latgridmin, gridmin
        
        
#%% Mixins for the localizers
class TDMixin:
    def run(self, s1x_list: np.ndarray, s2x_list: np.ndarray, tdoa_list: np.ndarray, td_sigma_list: np.ndarray):
        '''
        Performs TDOA weighted least squares error calculations on every point in the grid.
        TDOAs are assumed to be measured as (time to sensor 2) - (time to sensor 1).

        Parameters
        ----------
        s1x_list : np.ndarray
            K x 3 matrix. The positions (in units of metres) of the first sensor in Cartesian coordinates for every TDOA measurement.
        s2x_list : np.ndarray
            K x 3 matrix. The positions (in units of metres) of the second sensor in Cartesian coordinates for every TDOA measurement.
        tdoa_list : np.ndarray
            Length K array of TDOA measurements (in seconds).
        td_sigma_list : np.ndarray
            Length K array of TDOA measurement uncertainties (in seconds).

        Returns
        -------
        cost_grid : np.ndarray
            Length N array. The least squares errors for every grid point.
        '''
        if s1x_list.shape[1] != 3 or s2x_list.shape[1] != 3:
            raise ValueError("Ensure s1x_list & s2x_list have 3 columns.")

        cost_grid = gridSearchTDOA_direct(s1x_list, s2x_list, tdoa_list, td_sigma_list, self.gridmat)
        return cost_grid
    
#%%
class TDFDMixin:
    def run(self,
            s1x_list, s2x_list, tdoa_list, td_sigma_list,
            s1v_list, s2v_list, fdoa_list, fd_sigma_list, fc):
        '''
        Performs TDOA+FDOA weighted least squares error calculations on every point in the grid.
        TDOAs are assumed to be measured as (time to sensor 2) - (time to sensor 1).
        This convention holds for FDOA as well.

        Parameters
        ----------
        s1x_list : np.ndarray
            K x 3 matrix. The positions (in units of metres) of the first sensor in Cartesian coordinates for every measurement.
        s2x_list : np.ndarray
            K x 3 matrix. The positions (in units of metres) of the second sensor in Cartesian coordinates for every measurement.
        tdoa_list : np.ndarray
            Length K array of TDOA measurements (in seconds).
        td_sigma_list : np.ndarray
            Length K array of TDOA measurement uncertainties (in seconds).
        s1v_list : np.ndarray
            K x 3 matrix. The velocities (in units of m/s) of the first sensor in Cartesian coordinates for every measurement.
        s2v_list : np.ndarray
            K x 3 matrix. The velocities (in units of m/s) of the second sensor in Cartesian coordinates for every measurement.
        fdoa_list : np.ndarray
            Length K array of FDOA measurements (in Hz).
        fd_sigma_list : np.ndarray
            Length K array of FDOA measurement uncertainties (in Hz).
        fc : float
            Centre frequency. This is used to normalise the FDOA measurements.

        Returns
        -------
        cost_grid : np.ndarray
            Length N array. The least squares errors for every grid point.
        '''
        
        cost_grid = gridSearchTDFD_direct(s1x_list, s2x_list,
                                         tdoa_list, td_sigma_list,
                                         s1v_list, s2v_list,
                                         fdoa_list, fd_sigma_list, fc,
                                         self.gridmat, verb=True)
        return cost_grid
    
    def crb(self, gridmin: np.ndarray,
            s1x_list: np.ndarray, s2x_list: np.ndarray,
            s1v_list: np.ndarray, s2v_list: np.ndarray,
            td_sigma_list: np.ndarray, fd_sigma_list: np.ndarray, 
            fc: float):
        '''
        Calculates the raw CRB for TD+FD localization, assuming a certain position
        (this is usually extracted from the localization output) and a stationary target i.e. 0 velocity.
        Note, this also assumes a known altitude constraint (technically a known vector length constraint).

        Parameters
        ----------
        gridmin : np.ndarray
            The position to calculate the CRB around. In practical scenarios, we use the output from
            the localizer i.e. run() then localize().
        s1x_list : np.ndarray
            K x 3 matrix. Identical to run().
        s2x_list : np.ndarray
            K x 3 matrix. Identical to run().
        s1v_list : np.ndarray
            K x 3 matrix. Identical to run().
        s2v_list : np.ndarray
            K x 3 matrix. Identical to run().
        td_sigma_list : np.ndarray
            Length K array. Identical to run().
        fd_sigma_list : np.ndarray
            Length K array. Identical to run().
        fc : float
            Centre frequency. Identical to run().

        Returns
        -------
        crb : np.ndarray
            6 x 6 matrix. CRB output (with assumed constraints).
        '''
        
        S_combined = np.zeros((s1x_list.shape[0]*2,3))
        S_combined[0::2] = s2x_list # note the ordering is flipped here
        S_combined[1::2] = s1x_list
        S_combined = S_combined.T
        Sdot_combined = np.zeros((s1v_list.shape[0]*2,3))
        Sdot_combined[0::2] = s2v_list
        Sdot_combined[1::2] = s1v_list
        Sdot_combined = Sdot_combined.T

        cmat = np.vstack((
            np.hstack((gridmin, np.zeros(3))), # altitude constraint
            [0,0,0,1,0,0], # velocity component constraints
            [0,0,0,0,1,0],
            [0,0,0,0,0,1]
        )).T
        crb = calcCRB_TDFD(gridmin, S_combined, td_sigma_list*self.lightspd,
                            np.array([0,0,0]), Sdot_combined, fd_sigma_list/fc*self.lightspd, cmat=cmat)
        
        return crb
    
#%% Common combinations (remember that MRO is left to right)
class LatLonGridLocalizerTD(TDMixin, LatLonGridLocalizer):
    pass

class LatLonGridLocalizerTDFD(TDFDMixin, LatLonGridLocalizer):
    pass

#%% Satellite related routines and classes
def removeDownlinkRangeDiff(rx: np.ndarray, s1x: np.ndarray, s2x: np.ndarray, rangediff: np.ndarray):
    downlinkrangediff = rangeDifferenceOfArrival(rx, s1x, s2x)
    return rangediff - downlinkrangediff, downlinkrangediff

def removeDownlinkDopplerDiff(
    rx: np.ndarray, s1x: np.ndarray, s2x: np.ndarray,
    s1v: np.ndarray, s2v: np.ndarray, fdoa: float, f0down: float,
    lightspd: float=299792458.0
):
    rdot1 = calculateRangeRate(s1x, rx, tx_xdot=s1v)
    rdot2 = calculateRangeRate(s2x, rx, tx_xdot=s2v)
    downlinkFDOA = (-rdot2+rdot1)*f0down/lightspd
    return fdoa - downlinkFDOA, downlinkFDOA

    downlinkrangeratediff = rdot2 - rdot1
    return rangeratediff - downlinkrangeratediff, downlinkrangeratediff


class SatellitePairTDFDMixin(TDFDMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # Pass on everything

        # Add placeholders to check readiness
        self.prisat = None
        self.secsat = None
        self.rxecef = None

    @property
    def ready(self):
        """
        Checks if the RX position and the satellite objects have been set.
        """
        if self.rxecef is None or self.prisat is None or self.secsat is None:
            return False
        return True

    def configurePrimarySatellite(self, sat: skyfield.api.EarthSatellite):
        self.prisat = sat

    def configureSecondarySatellite(self, sat: skyfield.api.EarthSatellite):
        self.secsat = sat

    def setRXposition(self, rxecef: np.ndarray):
        self.rxecef = rxecef

    def _generateSatellitesPosVel(self, times: np.ndarray):
        # Generate geocentric class instances
        prigeocentrics = [sf_propagate_satellite_to_gpstime(self.prisat, time) for i in times]
        secgeocentrics = [sf_propagate_satellite_to_gpstime(self.secsat, time) for i in times]

        # Extract the position and velocity in ECEF (i.e. ITRS)
        pri_x_list = np.zeros((len(times),3))
        pri_xdot_list = np.zeros((len(times),3))
        sec_x_list = np.zeros((len(times),3))
        sec_xdot_list = np.zeros((len(times),3))
        for i in range(len(times)):
            pri_x, pri_xdot = sf_geocentric_to_itrs(prigeocentrics[i], returnVelocity=True)
            sec_x, sec_xdot = sf_geocentric_to_itrs(secgeocentrics[i], returnVelocity=True)
            # Place into giant matrix
            pri_x_list[i,:] = pri_x.m
            pri_xdot_list[i,:] = pri_xdot.m_per_s
            sec_x_list[i,:] = sec_x.m
            sec_xdot_list[i,:] = sec_xdot.m_per_s

        return pri_x_list, pri_xdot_list, sec_x_list, sec_xdot_list

    def run(self,
            times, tdoa_list, td_sigma_list,
            fdoa_list, fd_sigma_list, fc_up, fc_down):
        '''
        Performs TDOA+FDOA weighted least squares error calculations on every point in the grid.
        TDOAs are assumed to be measured as (time to sensor 2) - (time to sensor 1).
        This convention holds for FDOA as well.

        Parameters
        ----------
        times : np.ndarray
            Length K array of TDOA measurement times (in UTC seconds).
            This is used to propagate the satellites to their 
            correct positions/velocities at each measurement.
        tdoa_list : np.ndarray
            Length K array of TDOA measurements (in seconds).
        td_sigma_list : np.ndarray
            Length K array of TDOA measurement uncertainties (in seconds).
        fdoa_list : np.ndarray
            Length K array of FDOA measurements (in Hz).
        fd_sigma_list : np.ndarray
            Length K array of FDOA measurement uncertainties (in Hz).
        fc_up : float
            Centre frequency of the uplink.
        fc_down : float
            Centre frequency of the downlink.

        Returns
        -------
        cost_grid : np.ndarray
            Length N array. The least squares errors for every grid point.
        '''

        pri_x_list, pri_xdot_list, sec_x_list, sec_xdot_list = self._generateSatellitesPosVel(
            times
        )

        cost_grid = super().run(
            pri_x_list, sec_x_list, tdoa_list, td_sigma_list,
            pri_xdot_list, sec_xdot_list, fdoa_list, fd_sigma_list, fc_up)

        return cost_grid

        

        

        


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

