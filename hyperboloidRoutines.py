# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 17:52:48 2022

@author: Lken
"""

import numpy as np
from plotRoutines import *

#%% 
class Hyperboloid:
    def __init__(self, a: float, c: float,
                 mu: np.ndarray=np.zeros(3),
                 Rx: np.ndarray=np.eye(3),
                 Rz: np.ndarray=np.eye(3)):
        '''
        Defines a z-axis symmetric, two-sheet hyperboloid function
        generated from revolution where
        
        .. math::
            \frac{x^2}{a^2} + \frac{y^2}{a^2} - \frac{z^2}{c^2} = -1

        Parameters
        ----------
        a : float
            Constant for x, y.
        c : float
            Constant for z. This is determined by foci at +/- c.

        '''
        self.a = a
        self.c = c
        self.rangediff = c / 2
        self.focusZ = np.sqrt(a**2 + c**2)
        
        self.mu = mu
        self.Rx = Rx
        self.Rz = Rz
        self.Rot = self.Rz @ self.Rx # Convenient combined rotation
        
        # Pre-generate the foci
        foci = np.array([[0, 0, -self.focusZ],
                         [0, 0,  self.focusZ]])
        self.foci = np.zeros((3,2))
        self.foci[0,:], self.foci[1,:], self.foci[2,:] = self.transform(foci[:,0], foci[:,1], foci[:,2])
        
    # Parametrised equations
    def x(self, v, theta):
        return self.a * np.sinh(v) * np.cos(theta)
    
    def y(self, v, theta):
        return self.a * np.sinh(v) * np.sin(theta)
    
    def zplus(self, v):
        return self.c * np.cosh(v)
    
    def zminus(self, v):
        return -self.c * np.cosh(v)
    
    # Rotations and translations
    def transform(self, X, Y, Z):
        xshape = X.shape
        yshape = Y.shape
        zshape = Z.shape
        
        # Create 3-d vectors
        vecs = np.vstack((
                X.reshape(-1),
                Y.reshape(-1),
                Z.reshape(-1)
            ))
        
        # Transform
        vecs = (self.Rot @ vecs) + self.mu.reshape((-1,1))
        # return vecs
        
        X1 = vecs[0,:].reshape(xshape)
        Y1 = vecs[1,:].reshape(yshape)
        Z1 = vecs[2,:].reshape(zshape)
        
        return X1, Y1, Z1
    
    # Other methods    
    def visualize(self, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        
        
        # Generate some points in a mesh
        v = np.arange(0, 2, 0.1)
        theta = np.arange(-np.pi, np.pi, 0.1)
        v, theta = np.meshgrid(v, theta)
        
        # Calculate the cartesian coordinates from parametrisation
        X0 = self.x(v, theta)
        Y0 = self.y(v, theta)
        Zp0 = self.zplus(v)
        Zm0 = self.zminus(v)
        
        # Transform to actual orientation and position
        Xp, Yp, Zp = self.transform(X0, Y0, Zp0)
        Xm, Ym, Zm = self.transform(X0, Y0, Zm0)
        
        # Ensure equal ratios
        ax.set_box_aspect((
            np.ptp(np.hstack((Xp.reshape(-1), Xm.reshape(-1)))),
            np.ptp(np.hstack((Yp.reshape(-1), Ym.reshape(-1)))),
            np.ptp(np.hstack((Zp.reshape(-1), Zm.reshape(-1))))
        )) 
        
        
        # Plot both sheets
        ax.plot_wireframe(Xp, Yp, Zp) # In order to see the foci, use wireframe rather than surface
        ax.plot_wireframe(Xm, Ym, Zm)
        
        # Plot foci
        ax.scatter3D(self.foci[0,:], self.foci[1,:], self.foci[2,:], c='r')
        ax.set_title("Foci: (%.3f, %.3f, %.3f), (%.3f, %.3f, %.3f)" % (
            *self.foci[:,0], *self.foci[:,1])
        )
        
        return ax, fig # , Xp, Yp, Zp, Xm, Ym, Zm

        
        
    # Factory functions
    @classmethod
    def fromFoci(cls, s1: np.ndarray, s2: np.ndarray, rangediff: float):
        '''
        Parameters
        ----------
        s1 : np.ndarray
            Vector of 1st focus location.
        s2 : np.ndarray
            Vector of 1st focus location.
        rangediff : float
            Range difference of the hyperboloid sheet, with convention of 
            (s2 - x) - (s1 - x).
        '''
        
        # Connecting vector
        v = s2 - s1
        vnorm = np.linalg.norm(v)
        d = vnorm / 2
        
        # Need to produce rotation matrices that go from standard z-axis to one that is aligned with connecting vector
        # Rotate around X until theta
        theta = np.arccos(np.dot(v, np.array([0,0,1])) / vnorm)
        # theta = np.arctan((v[0]**2 + v[1]**2)**0.5 / v[2]) # sqrt(x^2 + y^2) / z
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(theta), -np.sin(theta)],
                       [0, np.sin(theta),  np.cos(theta)]])
        
        # Right-handed rotation results in vector pointing in reverse y-direction i.e.
        # phi0 = -pi/2
        # Rotate around z until pointing along phi1
        phi = np.arctan2(v[1], v[0]) + np.pi/2
        Rz = np.array([[np.cos(phi), -np.sin(phi), 0],
                       [np.sin(phi),  np.cos(phi), 0],
                       [0, 0, 1]])
        
        # # Debugging
        # print("theta = %f, phi = %f" % (theta, phi))
        # breakpoint()
        
        # Calculate the other parameters
        c = 0.5 * rangediff
        a = np.sqrt(d**2 - c**2)
        mu = (s2 + s1) / 2
        
        return cls(a, c, mu, Rx, Rz)
        
        
#%% Testing
if __name__ == "__main__":
    closeAllFigs()
    
    # Basic zero-centred default
    h = Hyperboloid(1, 0.1)
    ax, fig = h.visualize()
    
    # Generated in x-y plane
    hp = Hyperboloid.fromFoci(
        np.array([-1, 0, 0]),
        np.array([1, 0, 0]),
        0.5)
    ax, fig = hp.visualize()
    
    # Generated at arbitrary orientation, but still zero-centred
    ha = Hyperboloid.fromFoci(
        np.array([-1, -1, -1]),
        np.array([1, 1, 1]),
        1.0
    )
    ax, fig = ha.visualize()
    
    # Generated at arbitrary orientation at translated origin
    hat = Hyperboloid.fromFoci(
        10 + np.array([-1, -1, -1]),
        10 + np.array([1, 1, 1]),
        1.0
    )
    ax, fig = hat.visualize()
    
    # Generate reversed
    hatr = Hyperboloid.fromFoci(
        10 + np.array([1, 1, 1]),
        10 + np.array([-1, -1, -1]),
        -1.0
    )
    ax, fig = hatr.visualize() 
    
    #%% Unit tests
    import unittest
    class TestHyperboloids(unittest.TestCase):
        def test_zero_plane(self):
            np.testing.assert_allclose(
                hp.foci[:,0], [-1,0,0], atol=1e-7)
            np.testing.assert_allclose(
                hp.foci[:,1], [1,0,0], atol=1e-7)
            
        def test_zero_orient(self):
            np.testing.assert_allclose(
                ha.foci[:,0], [-1,-1,-1], atol=1e-7)
            np.testing.assert_allclose(
                ha.foci[:,1], [1,1,1], atol=1e-7)
            
        def test_trans_orient(self):
            np.testing.assert_allclose(
                hat.foci[:,0], [9,9,9], atol=1e-7)
            np.testing.assert_allclose(
                hat.foci[:,1], [11,11,11], atol=1e-7)
            
            
    unittest.main()
    