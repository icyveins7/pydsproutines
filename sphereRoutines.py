# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 15:40:43 2022

@author: Lken
"""

import numpy as np
from plotRoutines import *

#%%
class Ellipsoid:
    def __init__(self, a: float, b: float, c: float, 
                 mu: np.ndarray=np.zeros(3),
                 Rx: np.ndarray=np.eye(3),
                 Rz: np.ndarray=np.eye(3)):
        '''
        Container for a general ellipsoid, given by
        
        .. math::
            \frac{x^2}{a^2} + \frac{y^2}{b^2} + \frac{z^2}{c^2} = 1

        Parameters
        ----------
        a : float
            Constant for x.
        b : float
            Constant for y.
        c : float
            Constant for z.
        mu : 1-D array
            Translation vector i.e. position vector of centre.
        Rx : 2-D array
            X-axis rotation matrix.
        Rz : 2-D array
            Z-axis rotation matrix.

        '''
        
        self.a = a
        self.b = b
        self.c = c
        self.mu = mu
        self.Rx = Rx
        self.Rz = Rz
        
class OblateSpheroid(Ellipsoid):
    def __init__(self, omega: float, lmbda: float,
                 mu: np.ndarray=np.zeros(3),
                 Rx: np.ndarray=np.eye(3),
                 Rz: np.ndarray=np.eye(3)):
        
        self.omega = omega
        self.lmbda = lmbda
        assert(lmbda > omega)
        super().__init__(omega, omega, lmbda, mu, Rx, Rz)


class Sphere(Ellipsoid):
    def __init__(self, r: float, mu: np.ndarray=np.zeros(3)):
        self.r = r
        super().__init__(r, r, r, mu)
    
    def pointsFromAngles(self, theta, phi):
        points = np.array([
            self.r * np.sin(theta) * np.cos(phi),
            self.r * np.sin(theta) * np.sin(phi),
            self.r * np.cos(theta)
        ])    
        return points
    
    def transform(self, points):
        if points.ndim == 3:
            return points + self.mu.reshape((-1,1,1))
        else:
            return points + self.mu.reshape((-1,1))
        
    def intersectOblateSpheroid(self, theta, omega, lmbda):
        rs = self.r * np.sin(theta)
        rc = self.r * np.cos(theta)
        
        # Compute expanded coefficients
        gamma = lmbda**2 * (rs**2 + self.mu[0]**2 + self.mu[1]**2)
        beta = omega**2 * (rc**2 + 2 * rc * self.mu[2] + self.mu[2]**2)
        A = lmbda**2 * 2 * rs * self.mu[0]
        B = lmbda**2 * 2 * rs * self.mu[1]
        
        with np.errstate(divide='ignore', invalid='ignore'):
            alpha = np.arctan(B/A)
            t = (lmbda**2 * omega**2 -beta-gamma)/np.sqrt(A**2 + B**2)
            # breakpoint()
            basic = np.arccos(t) # returns [0, pi]
        
        # Remove nans
        idx = ~np.isnan(basic)
        basic = basic[idx]
        alpha = alpha[idx]
        theta = theta[idx]
        
        # Find both quadrants
        phi1 = basic + alpha
        phi2 = -basic + alpha
        
        phi = np.hstack((phi1[::-1], phi2))
        thetae = np.hstack((theta[::-1], theta))
        
        points = self.pointsFromAngles(thetae, phi)
        points = self.transform(points)
        
        return points
    
    # Other methods    
    def visualize(self, theta=np.arange(0,np.pi,0.001), ax=None, bothSheets=False, useSurf=False):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        
        
        # Generate some points in a mesh
        phi = np.arange(-np.pi, np.pi, 0.1)
        theta, phi = np.meshgrid(theta, phi)
        
        # Calculate the cartesian coordinates from parametrisation
        points = self.pointsFromAngles(theta, phi)
        # breakpoint()
        points = self.transform(points)
        
        # Ensure equal ratios
        ax.set_box_aspect((
            np.ptp(points[0].reshape(-1)),
            np.ptp(points[1].reshape(-1)),
            np.ptp(points[2].reshape(-1))
        )) 
        
        ax.plot_wireframe(points[0], points[1], points[2], color='k')
        
        return ax, fig
    
    
#%% Testing
if __name__ == "__main__":
    closeAllFigs()
    sphere = Sphere(1, np.array([1,2,0]))
    
    theta = np.arange(0, np.pi, 0.01)
    omega = 2.0
    lmbda = 1.5
    points = sphere.intersectOblateSpheroid(theta, omega, lmbda)
    
    ax, fig = sphere.visualize()
    # Create spheroid
    theta = np.arange(0, np.pi, 0.1)
    phi = np.arange(0, 2*np.pi, 0.1)
    theta, phi = np.meshgrid(theta, phi)
    x = omega * np.sin(theta) * np.cos(phi)
    y = omega * np.sin(theta) * np.sin(phi)
    z = lmbda * np.cos(theta)
    ax.plot_wireframe(x,y,z,linestyle='--')
    ax.set_box_aspect(None)
    ax.plot3D(points[0], points[1], points[2], 'r-')
    
    # Check if points truly lie on surface
    check = points[0,:]**2/omega**2 + points[1,:]**2/omega**2 + points[2,:]**2/lmbda**2