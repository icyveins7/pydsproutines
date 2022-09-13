# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 17:52:48 2022

@author: Lken
"""

import numpy as np

#%% 
class Hyperboloid:
    def __init__(self, a: float, c: float, mu: np.ndarray=None, Rx: np.ndarray=None, Ry: np.ndarray=None):
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
        self.Rx = Rx
        if self.Rx is None:
            self.Rx = np.eye(3)
        self.Ry = Ry
        if self.Ry is None:
            self.Ry = np.eye(3)
        
    @classmethod
    def fromFoci(cls, s1: np.ndarray, s2: np.ndarray, rangediff: float):
        # Connecting vector
        v = s2 - s1
        d = np.linalg.norm(v) / 2
        # v = v / (2*d) # Normalise?
        
        # Need to produce rotation matrices that go from standard z-axis to one that is aligned with connecting vector
        # Rotate around X until theta
        
        # Rotate around z, until above x-axis
        rhoZ = np.arctan2(v[1], v[0]) # This is -
        Rz = np.array([[ np.cos(rhoZ), np.sin(rhoZ), 0],
                       [-np.sin(rhoZ), np.cos(rhoZ), 0],
                       [0, 0, 1]])
        v = Rz @ v
        
        # Rotate around y, until parallel to z
        rhoY = np.arctan2(v[1], v[2]) # This is -
        Ry = np.array([[ np.cos(rhoY), 0, np.sin(rhoY)],
                       [0, 1, 0],
                       [-np.sin(rhoY), 0, np.cos(rhoY)]])
        v = Ry @ v
        
        c = 0.5 * rangediff
        a = np.sqrt(d**2 - c**2)
        
        
#%% Testing
if __name__ == "__main__":
    pass
    