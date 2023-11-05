# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 17:52:48 2022

@author: Lken
"""

import numpy as np
from plotRoutines import *
# from numba import njit

from timingRoutines import Timer

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
        mu : 1-D array
            Translation vector i.e. position vector of centre of foci.
        Rx : 2-D array
            X-axis rotation matrix.
        Rz : 2-D array
            Z-axis rotation matrix.

        '''
        self.a = a
        self.c = c
        # convention here is for c to be the same sign as rangediff
        # Note, this means that negative c -> use the upper sheet, positive c -> use the lower sheet
        # However in practice, we can just use the sign of c as the indicator -> always use the (-c) coefficient as the correct sheet
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
    
    def z(self, v, sign):
        if sign > 0:
            return self.zplus(v)
        else:
            return self.zminus(v)
    
    # Rotations and translations
    def transform(self, X, Y, Z):
        xshape = X.shape
        yshape = Y.shape
        zshape = Z.shape
        
        # Create 3-d vectors, each vector is now a column
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
    
    def inverseTransform(self, points: np.ndarray):
        assert(points.ndim == 2 and points.shape[0] == 3) # Enforce shape
        # Broadcast the inverse translation
        vecs = points - self.mu.reshape((-1,1))
        vecs = np.linalg.inv(self.Rot) @ vecs
        return vecs
    
    # Other methods    
    def visualize(self, v=np.arange(0, 2, 0.1), ax=None, bothSheets=False, useSurf=False):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        
        
        # Generate some points in a mesh
        # v = np.arange(0, 2, 0.1)
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
        self.visxlim = np.ptp(np.hstack((Xp.reshape(-1), Xm.reshape(-1)))) # We keep these around for future reference
        self.visylim = np.ptp(np.hstack((Yp.reshape(-1), Ym.reshape(-1)))) # in order to plot other objects around to scale together
        self.viszlim = np.ptp(np.hstack((Zp.reshape(-1), Zm.reshape(-1))))
        ax.set_box_aspect((
            self.visxlim,
            self.visylim,
            self.viszlim
        )) 
        
        
        # Plot both sheets
        if useSurf:
            if bothSheets:
                ax.plot_surface(Xp, Yp, Zp, cmap='viridis')
            ax.plot_surface(Xm, Ym, Zm, cmap='viridis')
        else:
            if bothSheets:
                ax.plot_wireframe(Xp, Yp, Zp, color='k', linestyle='--') # In order to see the foci, use wireframe rather than surface
            ax.plot_wireframe(Xm, Ym, Zm, color='k', linestyle='-')
        
        # Plot foci
        ax.scatter3D(self.foci[0,:], self.foci[1,:], self.foci[2,:], c='r')
        ax.set_title("Foci: (%.3f, %.3f, %.3f), (%.3f, %.3f, %.3f)" % (
            *self.foci[:,0], *self.foci[:,1])
        )
        
        return ax, fig # , Xp, Yp, Zp, Xm, Ym, Zm
    
    # Intersection Methods
    def _intersectXYsheet(self, v, sign):
        sinhv = np.sinh(v)
        coshv = np.cosh(v)
        A_0 = self.Rot[2,0] * self.a * sinhv
        A_1 = self.Rot[2,1] * self.a * sinhv
        A_2 = self.Rot[2,2] * sign * self.c * coshv + self.mu[2]
        
        with np.errstate(divide='ignore', invalid='ignore'):
            alpha = np.arctan(A_0 / A_1)
        
            b = -A_2 / np.sqrt(A_0**2 + A_1**2)
        
            theta1 = np.arcsin(b) # If there are NaNs here, it means that the particular v does not have a solution!
            theta2 = np.sign(b) * np.pi - theta1
  
        theta = np.hstack((theta2[::-1], theta1)) - np.hstack((alpha[::-1], alpha))
        v_ext = np.hstack((v[::-1], v))
        
        x = self.x(v_ext, theta)
        y = self.y(v_ext, theta)
        z = self.z(v_ext, sign)
        
        # Remove nans?
        idx = np.logical_and(np.logical_and(~np.isnan(x), ~np.isnan(y)), ~np.isnan(z))
        x = x[idx]
        y = y[idx]
        z = z[idx]
        
        vec = np.vstack((
            x,
            y,
            z
        ))
        
        # Perform the transformation
        vect = np.zeros_like(vec)   
        vect[0,:], vect[1,:], vect[2,:] = self.transform(vec[0,:], vec[1,:], vec[2,:])
        
        return vect

        
    
    def intersectXY(self, v=np.arange(0, 2, 0.01), onlyReturnOneSheet=False):
        msheet = self._intersectXYsheet(v, -1) # This is the main sheet which corresponds correctly to range diff
        if onlyReturnOneSheet:
            return msheet
        else:
            psheet = self._intersectXYsheet(v, 1)
            return msheet, psheet
        
    @staticmethod  
    # @njit('Tuple((float64[:], float64[:]))(float64[:,:], float64[:])', nogil=True)
    def _intersectOblateSpheroidLoop(tc, v):
        thetarealplus = list()
        thetarealminus = list()
        veplus = list()
        veminus = list()
        
        for i in np.arange(tc.shape[1]):
            coeffs = tc[::-1, i] # In descending polynomial order
            
            # Descartes rule of signs
            # We have 4 roots, and only real coefficients, so any complex roots come in pairs
            # Hence there's either 2 real roots (or repeated real roots) and 2 complex roots,
            # or just 4 complex roots (invalid answer for us)
            # As such we can determine the number of roots before we even decide to perform the root finder,
            # which is expensive
            # We do this by checking the sign changes are an odd number, which would suggest at least 1 positive root.
            # There is a slim chance that there are 2 positive roots, or 2 negative roots, which would be ignored under this scheme,
            # but that is very unlikely given the geometry of the problem
            descartes = False
            signchanges = np.diff(np.sign(coeffs))
            if np.argwhere(np.diff(np.sign(coeffs))).size % 2 != 0:
                descartes = True

                roots = np.roots(coeffs) # .astype(np.complex128))
                # Casting it to complex128 introduced rounding errors (extra small imaginary components)
                # so why did i do this at first??
                            
                ## Instead of checking sign, sort them (this accounts for two negative or two positive roots, which is possible)
                thetas = np.arctan(roots) * 2
                thetas = np.real(thetas[np.imag(thetas) == 0]) # Extract only real roots
                thetas = np.sort(thetas) # sort them
                
                # Make the assumption that there are only up to 2 roots (which there should only be)
                if thetas.size >= 1:
                    # Simply append to the negatives
                    thetarealminus.insert(0,thetas[0])
                    veminus.insert(0,v[i])
                if thetas.size == 2:
                    # If more than one, we push the second one to the positives, since it's sorted already
                    thetarealplus.append(thetas[1])
                    veplus.append(v[i])
                    
                    
        return thetarealminus, veminus, thetarealplus, veplus
    
    def _estimateSpheroidV(self, omega, lmbda):
        # First get the midpoint of the foci
        fociMid = np.mean(self.foci, axis=1) # 1-d row vector
        # Now estimate the v required to reach the centre, as a gauge
        # First project the origin into the hyperboloid coordinate space
        pzero = self.inverseTransform(np.zeros(3).reshape((3,-1)))
        # We use x and y to calculate v via
        vmid = np.arcsinh(np.sqrt(np.sum(pzero[:2]**2) / self.a**2))
        # Estimate the bounds by just moving outwards by the larger of the two spheroid constants
        outer = np.max([omega, lmbda]) * fociMid / np.linalg.norm(fociMid)
        pouter = self.inverseTransform(outer.reshape((3,-1)))
        vout = np.arcsinh(np.sqrt(np.sum(pouter[:2]**2) / self.a**2))
        
        return vout, vmid
        
    def _generateIntersectOblateSpheroidCoefficients(self, v, omega, lmbda):
        sinhv = np.sinh(v)
        coshv = np.cosh(v)
        a_sinhv = self.a * sinhv
        mc_coshv = -self.c * coshv
        # X components
        l_0 = self.Rot[0,0] * a_sinhv
        l_1 = self.Rot[0,1] * a_sinhv
        l_2 = self.Rot[0,2] * mc_coshv + self.mu[0] # For now, let's just work on the correct sheet
        # Y components
        m_0 = self.Rot[1,0] * a_sinhv
        m_1 = self.Rot[1,1] * a_sinhv
        m_2 = self.Rot[1,2] * mc_coshv + self.mu[1] # For now, let's just work on the correct sheet
        # Z components
        n_0 = self.Rot[2,0] * a_sinhv
        n_1 = self.Rot[2,1] * a_sinhv
        n_2 = self.Rot[2,2] * mc_coshv + self.mu[2] # For now, let's just work on the correct sheet
        
        # Compressed coefficients for t-substitution
        alpha = np.vstack((
            l_0**2 + 2 * l_0 * l_2 + l_2**2,
            4 * l_0 * l_1 + 4 * l_1 * l_2,
            -2 * l_0**2 + 4 * l_1**2 + 2 * l_2**2,
            -4 * l_0 * l_1 + 4 * l_1 * l_2,
            l_0**2 - 2 * l_0 * l_2 + l_2**2
        ))
        beta = np.vstack((
            m_0**2 + 2 * m_0 * m_2 + m_2**2,
            4 * m_0 * m_1 + 4 * m_1 * m_2,
            -2 * m_0**2 + 4 * m_1**2 + 2 * m_2**2,
            -4 * m_0 * m_1 + 4 * m_1 * m_2,
            m_0**2 - 2 * m_0 * m_2 + m_2**2
        ))
        gamma = np.vstack((
            n_0**2 + 2 * n_0 * n_2 + n_2**2,
            4 * n_0 * n_1 + 4 * n_1 * n_2,
            -2 * n_0**2 + 4 * n_1**2 + 2 * n_2**2,
            -4 * n_0 * n_1 + 4 * n_1 * n_2,
            n_0**2 - 2 * n_0 * n_2 + n_2**2
        ))
        
        # Coefficients for polynomial
        tc = (alpha + beta) * lmbda**2 + gamma * omega**2
        # Corrections
        osqlsq = omega**2 * lmbda**2
        tc[0,:] = tc[0,:] - osqlsq
        tc[2,:] = tc[2,:] - 2 * osqlsq
        tc[4,:] = tc[4,:] - osqlsq
        
        return tc
        
    def intersectOblateSpheroid(
            self,
            v: np.ndarray=None, 
            omega: float=6378137.0,
            lmbda: float=6356752.314245,
            numPts: int=100,
            refineMiddle: bool=True):
        '''
        Oblate spheroid (ellipsoid generated by revolution around z-axis) intersection.

        Parameters
        ----------
        omega : float
            Constant for semi-major axis (x-y).
        lmbda : float
            Constant for semi-minor axis (z).

        Returns
        -------
        None.

        '''
        
        # Generate a reasonable set of v points if not provided
        if v is None:
            vout, vmid = self._estimateSpheroidV(omega, lmbda)
            v = np.linspace(0.9*vout, vmid, numPts)
        
        
        tc = self._generateIntersectOblateSpheroidCoefficients(v, omega, lmbda)
        
        # Numba-compiled method for speed over the loops
        thetarealminus, veminus, thetarealplus, veplus = self._intersectOblateSpheroidLoop(tc, v)
        
        # After the first run, get the smallest v value with real roots
        if refineMiddle:
            vspace = veplus[1]-veplus[0]
            vext = np.linspace(veplus[0]-vspace, veplus[0], numPts//2, endpoint=False) # Split within the nearest step, with fewer steps required
            
            # Do it all again
            tcext = self._generateIntersectOblateSpheroidCoefficients(vext, omega, lmbda)
            ethetarealminus, eveminus, ethetarealplus, eveplus = self._intersectOblateSpheroidLoop(tcext, vext)
        
            # Stitch lists
            nthetareals = np.hstack((thetarealminus, ethetarealminus, ethetarealplus, thetarealplus))
            nve = np.hstack((veminus, eveminus, eveplus, veplus))
            
        else:
            nthetareals = np.hstack((thetarealminus, thetarealplus))
            nve = np.hstack((veminus, veplus))
        
        ptx = self.x(nve, nthetareals)
        pty = self.y(nve, nthetareals)
        ptz = self.z(nve, -1)
        tpoints = np.zeros((3,nthetareals.size))
        tpoints[0,:], tpoints[1,:], tpoints[2,:] = self.transform(ptx,pty,ptz)
        
        
        return tpoints, nve

        
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
        c = 0.5 * rangediff # c takes the convention of the same sign as range diff
        a = np.sqrt(d**2 - c**2)
        mu = (s2 + s1) / 2
        
        return cls(a, c, mu, Rx, Rz)
        
        
#%% Testing
if __name__ == "__main__":
    from timingRoutines import Timer
    closeAllFigs()
    timer = Timer()
    
    # Basic zero-centred default
    h = Hyperboloid(1, 0.1)
    ax, fig = h.visualize()
    
    # Generated in x-y plane
    hp = Hyperboloid.fromFoci(
        np.array([-1, 0, 0]),
        np.array([1, 0, 0]),
        -1.0)
    ax, fig = hp.visualize()
    # Perform intersection
    vecs_m, vecs_p = hp.intersectXY()
    ax.plot3D(vecs_p[0,:], vecs_p[1,:], vecs_p[2,:], 'b-')
    ax.plot3D(vecs_m[0,:], vecs_m[1,:], vecs_m[2,:], 'g-')
    plt.figure()
    plt.plot(vecs_p[0], vecs_p[1], 'x-')
    plt.plot(vecs_m[0], vecs_m[1], 'x-')
    
    
    # Generated at arbitrary orientation, but still zero-centred
    ha = Hyperboloid.fromFoci(
        np.array([-1, -1, -1]),
        np.array([1, 1, 1]),
        1.0
    )
    ax, fig = ha.visualize()
    # Perform intersection
    vecs_m, vecs_p = ha.intersectXY()
    ax.plot3D(vecs_p[0,:], vecs_p[1,:], vecs_p[2,:], 'b-')
    ax.plot3D(vecs_m[0,:], vecs_m[1,:], vecs_m[2,:], 'g-')
    plt.figure()
    plt.plot(vecs_p[0], vecs_p[1], 'x-')
    plt.plot(vecs_m[0], vecs_m[1], 'x-')
    
    # Generated at arbitrary orientation at translated origin
    hat = Hyperboloid.fromFoci(
        10 + np.array([-1, -1, -1]),
        10 + np.array([1, 1, 1]),
        1.0
    )
    ax, fig = hat.visualize(v=np.arange(0,3.5,0.1))
    # Perform intersection
    timer.start()
    vecs_m, vecs_p = hat.intersectXY(v=np.arange(0,3.5,0.001))
    timer.end()
    ax.plot3D(vecs_p[0,:], vecs_p[1,:], vecs_p[2,:], 'b-')
    ax.plot3D(vecs_m[0,:], vecs_m[1,:], vecs_m[2,:], 'g-')
    plt.figure()
    plt.plot(vecs_p[0], vecs_p[1], 'x-')
    plt.plot(vecs_m[0], vecs_m[1], 'x-')
    
    
    
    # Generate for a spheroid
    hsp = Hyperboloid.fromFoci(
        np.array([-10, 1, 0]),
        np.array([-10, -1, 0]),
        0.5
    )
    vs = np.arange(1.5,3,0.001)
    ax, fig = hsp.visualize(vs)
    # Create spheroid
    theta = np.arange(0, np.pi, 0.1)
    phi = np.arange(0, 2*np.pi, 0.1)
    theta, phi = np.meshgrid(theta, phi)
    omega = 5.0 # Controls xy
    lmbda = 4.0 # Controls z
    x = omega * np.sin(theta) * np.cos(phi)
    y = omega * np.sin(theta) * np.sin(phi)
    z = lmbda * np.cos(theta)
    ax.plot_wireframe(x,y,z,linestyle='--')
    ax.set_box_aspect(None)
    # Attempt to find intersection points
    timer.start()
    tpts, ve = hsp.intersectOblateSpheroid(None, omega, lmbda) # vs,omega,lmbda)
    timer.end("intersectOblateSpheroid, %d pts" % (vs.size))
    ax.plot3D(tpts[0,:],tpts[1,:],tpts[2,:],'r')
    # Check if points truly lie on surface
    check = tpts[0,:]**2/omega**2 + tpts[1,:]**2/omega**2 + tpts[2,:]**2/lmbda**2
    
    
    ### Generate a typical satellite
    from satelliteRoutines import *
    from localizationRoutines import *
    sat1 = Satellite(
        '1 42691U 17023A   23217.40909002 -.00000373  00000+0  00000+0 0  9996',
        '2 42691   0.0264  36.5306 0000462  83.0552  97.2787  1.00273009 22943',
        name='KOREASAT 7', const=WGS84
    )
    gc1 = sf_propagate_satellite_to_gpstime(sat1, 1691227819.0)
    satecef1 = sf_geocentric_to_itrs(gc1).m
    satlla1 = ecef2geodeticLLA(satecef1).reshape(-1)
    
    sat2 = Satellite(
        "1 29349U 06034A   23217.40856704 -.00000369  00000+0  00000+0 0  9996",
        "2 29349   0.0242 184.9282 0001623 321.2519  67.6247  1.00273151 62130",
        "KOREASAT 5", const=WGS84
    )
    gc2 = sf_propagate_satellite_to_gpstime(sat2, 1691227819.0)
    satecef2 = sf_geocentric_to_itrs(gc2).m
    satlla2 = ecef2geodeticLLA(satecef2).reshape(-1)
    
    # Generate the hyperboloid for these 2
    from scipy.constants import speed_of_light
    hsat = Hyperboloid.fromFoci(satecef1, satecef2, 0) # -1e-5 * speed_of_light)
    timer.start()
    sathyp, sve = hsat.intersectOblateSpheroid(numPts=1000) # We have now adapted the algo to generate points better at the centre
    timer.end("intersectOblateSpheroid, %d pts" % (1000))
    sathyplla = ecef2geodeticLLA(sathyp.T)
    satfig, satax = plt.subplots(1,1)
    satax.plot(sathyplla[:,1], sathyplla[:,0], 'x-')
    satax.plot(satlla1[1], satlla1[0], 'kx')
    satax.plot(satlla2[1], satlla2[0], 'rx')
    
    
    
    plt.show()
    
    
    #%% Unit tests
    import unittest
    class TestHyperboloids(unittest.TestCase):
        def test_zero_plane(self):
            # Test the foci locations
            np.testing.assert_allclose(
                hp.foci[:,0], [-1,0,0], atol=1e-7)
            np.testing.assert_allclose(
                hp.foci[:,1], [1,0,0], atol=1e-7)
            # Test the sheet correctness
            vm = hp.intersectXY(onlyReturnOneSheet=True)
            np.testing.assert_allclose(
                np.linalg.norm(vm - hp.foci[:,1].reshape((-1,1)), axis=0) - np.linalg.norm(vm - hp.foci[:,0].reshape((-1,1)), axis=0),
                np.zeros(vm.shape[1]) - 1.0
            )
            
            
        def test_zero_orient(self):
            # Test the foci locations
            np.testing.assert_allclose(
                ha.foci[:,0], [-1,-1,-1], atol=1e-7)
            np.testing.assert_allclose(
                ha.foci[:,1], [1,1,1], atol=1e-7)
            # Test the sheet correctness
            vm = ha.intersectXY(onlyReturnOneSheet=True)
            np.testing.assert_allclose(
                np.linalg.norm(vm - ha.foci[:,1].reshape((-1,1)), axis=0) - np.linalg.norm(vm - ha.foci[:,0].reshape((-1,1)), axis=0),
                np.zeros(vm.shape[1]) + 1.0
            )
            
        def test_trans_orient(self):
            # Test the foci locations
            np.testing.assert_allclose(
                hat.foci[:,0], [9,9,9], atol=1e-7)
            np.testing.assert_allclose(
                hat.foci[:,1], [11,11,11], atol=1e-7)
            # Test the sheet correctness
            vm = hat.intersectXY(onlyReturnOneSheet=True)
            np.testing.assert_allclose(
                np.linalg.norm(vm - hat.foci[:,1].reshape((-1,1)), axis=0) - np.linalg.norm(vm - hat.foci[:,0].reshape((-1,1)), axis=0),
                np.zeros(vm.shape[1]) + 1.0
            )
            
        def test_random_hyperboloid(self):
            s = np.random.rand(3,2)
            rangediff = np.linalg.norm(s[:,1] - s[:,0]) * (np.random.rand() * 2 - 1) # Randomly choose either one
            hr = Hyperboloid.fromFoci(s[:,0], s[:,1], rangediff)
            # Generate the correct sheet
            vm = hr.intersectXY(onlyReturnOneSheet=True)
            np.testing.assert_allclose(
                np.linalg.norm(vm - hr.foci[:,1].reshape((-1,1)), axis=0) - np.linalg.norm(vm - hr.foci[:,0].reshape((-1,1)), axis=0),
                np.zeros(vm.shape[1]) + rangediff
            )
            
        def test_random_hyperboloid_spheroid(self):
            s = np.random.rand(3,2) + 1.0 # Move it outside
            rangediff = np.linalg.norm(s[:,1] - s[:,0]) * (np.random.rand() * 2 - 1) # Randomly choose either one
            hr = Hyperboloid.fromFoci(s[:,0], s[:,1], rangediff)
            # Generate the correct sheet
            omega = 1.0
            lmbda = 0.9
            v = np.arange(1.5, 3, 0.01)
            tpts, ve = hr.intersectOblateSpheroid(v, omega, lmbda)
            np.testing.assert_allclose(
                (tpts[0]**2 + tpts[1]**2) / (omega**2) + tpts[2]**2 / (lmbda**2),
                np.ones(tpts.shape[1])
            )
            
            
            
            
    # unittest.main()
    