import numpy as np
import scipy as sp
from scipy.constants import speed_of_light


class LocalizationCRBComponent:
    def __init__(
        self,
        x: np.ndarray,
        inv_sigma_sq: float | np.ndarray,
        S: np.ndarray
    ):
        """
        A general CRB component for localization.
        This usually corresponds to a single localization measurement.
        Generally speaking, this component is not directly constructed;
        see the subclasses instead.

        Note: in the event that multiple measurement scalars are correlated,
        all these scalars should be within 1 subclass. The sigma argument should then 
        necessarily be supplied as a matrix of shape (numScalars, numScalars).

        Otherwise, it is sufficient to split each measurement scalar into separate subclasses,
        and have each subclass use a scalar float sigma argument.

        For subclasses, the only required methods to reimplement are:
        1) Constructor __init__ itself (call super().__init__ if nothing else)
        2) _differentiate().

        Anything else is secondary/optional.

        Parameters
        ----------
        x : np.ndarray
            Target position vector.
        inv_sigma_sq : float or np.ndarray
            Related uncertainty for this measurement.
            Use a scalar float for a diagonal covariance matrix i.e. non-correlated measurements.
        S : np.ndarray
            m x 3 vectors, representing each sensor's position.
        """
        if x.shape != (3,):
            raise ValueError("x must be a length 3, 1D vector i.e shape (3,)")
        self.x = x
        if not isinstance(inv_sigma_sq, np.ndarray) and not isinstance(inv_sigma_sq, float): 
            raise TypeError("inv_sigma_sq must be a scalar float if not an array")
        self.inv_sigma_sq = inv_sigma_sq
        self.S = S

        # Storage for the partial derivatives vector
        self.partials = self._differentiate()

    def _differentiate(self):
        """
        Redefine this in subclass to properly differentiate the specific measurement type
        with respect to the target position vector.

        The subclassed method should return an np.ndarray of an appropriate shape.
        Some measurements may include two or more partial derivatives at once, which should
        be stacked using vstack(). 

        Raises
        ------
        NotImplementedError
            Throws this error if this method is not implemented in the subclass.
        """
        raise NotImplementedError("_differentiate() only implemented in subclasses.")
    
    def fim(self) -> np.ndarray:
        """
        Calculates and returns the FIM for this measurement/component.
        The calculation is (partial)^T * inv_sigma_sq * (partial).

        Returns
        -------
        fim : np.ndarray
            3x3 matrix of the FIM.
        """
        # Create the row vectors (last matrix)
        J = self.partials.reshape((-1,3)) # This is in case it's ndim=1

        if isinstance(self.inv_sigma_sq, np.ndarray):
            return J.T @ self.inv_sigma_sq.T @ J
        else:
            return J.T @ J * self.inv_sigma_sq
    

class AOA3DCRBComponent(LocalizationCRBComponent):
    def __init__(
        self,
        x: np.ndarray,
        delta: float,
        S: np.ndarray 
    ):
        """
        CRB component for angle of arrival in 3D, which corresponds to
        2 measurement angles: theta (elevation) and phi (azimuth).
        The coordinate system is such that theta is the angle away from the x-y plane;
        theta = (-pi/2, pi/2).

        Parameters
       ----------
        x : np.ndarray
            Target position vector.
        delta : float
            The mean squared angular error for this measurement, in radians.
            This subclass assumes an isotropic angular error (cone
            around the direction vector). The sigma_phi and sigma_theta values
            are back-calculated from this value.
        S : np.ndarray
            Length 3, 1D vector, representing the sensor's position. 
 
        Raises
        ------
        ValueError
            Thrown if S is not a length 3, 1D vector i.e. shape (3,)
        """
        # Create the direction vector before performing differentials
        if S.shape != (3, ):
            raise ValueError("S must be a length 3, 1D vector i.e shape (3,)")

        self.uf = x - S # Original direction vector
        self.u = self.uf / np.linalg.norm(self.uf) # Normalised direction vector

        # Calculate the direction vector's phi and theta
        self.phi = np.arctan2(self.u[1], self.u[0])
        self.theta = np.arcsin(self.u[2])

        # Calculate the sigma_phi and sigma_theta values
        self.delta = delta
        sigma_theta_sq = delta**2 / 2
        sigma_phi_sq = delta**2 / (2 * np.cos(self.theta)**2)

        # Perform differentials by calling the parent class
        super().__init__(x, 
                         np.array([[1/sigma_phi_sq, 0],[0, 1/sigma_theta_sq]]), 
                         S)

    @property
    def dphi(self) -> np.ndarray:
        """Extracts the partial derivatives of phi from self.partials."""
        return self.partials[0]
    
    @property
    def dtheta(self) -> np.ndarray:
        """Extracts the partial derivatives of theta from self.partials."""
        return self.partials[1]

    def _differentiate(self):
        """
        Calculate the partial derivatives of phi and theta into self.partials, in that order.

        Returns
        -------
        np.ndarray
            2 x 3 array with dphi and dtheta partial derivatives.
        """
        # Some small optimizations
        x2y2 = self.uf[0]**2 + self.uf[1]**2 # Used a lot below
        
        # Calculate phi derivatives
        dphi = np.zeros(3, np.float64)
        dphi[0] = -self.uf[1] / (x2y2)
        dphi[1] = self.uf[0] / (x2y2)
        dphi[2] = 0.0

        # Calculate theta derivatives
        dtheta = np.zeros(3, np.float64)
        dtheta[0] = - self.uf[2]*self.uf[0] / (np.linalg.norm(self.uf)**2 * np.sqrt(x2y2))
        dtheta[1] = - self.uf[2]*self.uf[1] / (np.linalg.norm(self.uf)**2 * np.sqrt(x2y2))
        dtheta[2] = np.sqrt(x2y2) / np.linalg.norm(self.uf)**2

        return np.vstack((dphi, dtheta))


class TDOACRBComponent(LocalizationCRBComponent):
    def __init__(
        self,
        x: np.ndarray,
        inv_sigma_td_sq: float,
        S: np.ndarray
    ):
        """
        Instantiate a TDOA CRB component.
        This corresponds to a single uncorrelated TDOA measurement between 2 sensors.

        Parameters
        ----------
        x : np.ndarray
            Target position vector.
        inv_sigma_td_sq : float
            Inverse uncertainty for the TDOA measurement, in units of s^-2.
            This will be internally converted to the appropriate range units required.
        S : np.ndarray
            A (2,3) shape matrix, representing the 2 sensors' positions;
            the TDOA measurement convention is assumed to be |x-S[1]| - |x-S[0]|.
        """
        # Check that there are 2 sensors exactly
        if S.shape != (2, 3):
            raise ValueError("S must be a (2,3) shape matrix")

        # Translate TDOA inv sigma value to RDOA inv sigma value
        self.inv_sigma_rdoa_sq = inv_sigma_td_sq/speed_of_light**2

        self.r = np.linalg.norm(x - S, axis=1) # This is the ranges to each sensor, length 2
        super().__init__(x, self.inv_sigma_rdoa_sq, S)

    def _differentiate(self) -> np.ndarray:
        """
        Calculate the partial derivatives of the range-difference.

        Returns
        -------
        np.ndarray
            (3, ) shape array for the partial derivatives.
        """
        # Calculate the partials for the individual ranges
        r_dx = (self.x - self.S) / self.r.reshape((-1,1)) # Need to reshape to column to broadcast divides
        
        # Return the difference of the partials
        return r_dx[1] - r_dx[0]


class TOACRBComponent(LocalizationCRBComponent):
    def __init__(self, x: np.ndarray, inv_sigma_tau_sq: float, S: np.ndarray):
        """
        Instantiate a TOA CRB component.
        This corresponds to a single uncorrelated TOA measurement from 1 sensor.

        Parameters
        ----------
        x : np.ndarray
            Target position vector.
        inv_sigma_tau_sq : float
            Inverse uncertainty for the TOA measurement, in units of s^-2.
            This will be internally converted to the appropriate range units required.
        S : np.ndarray
            A (3,) shape matrix, representing the sensor's position
        """
        # Check that there is exactly 1 sensor
        if S.shape != (3,):
            raise ValueError("S must be a (3,) shape matrix")

        # Translate TDOA inv sigma value to RDOA inv sigma value
        self.inv_sigma_roa_sq = inv_sigma_tau_sq/speed_of_light**2

        self.r = np.linalg.norm(x - S)
        super().__init__(x, self.inv_sigma_roa_sq, S)

    def _differentiate(self):
        """
        Calculate the partial derivatives of the range.

        Returns
        -------
        np.ndarray
            (3, ) shape array for the partial derivatives.
        """
        # Partial for range
        r_dx = (self.x - self.S) / self.r

        # Return as is
        return r_dx

 

#%%
class CRB:
    def __init__(self, constraints: np.ndarray=None):
        """
        Container for CRB components.
        Instantiate this and then use addComponent()
        on the individual components before using
        compute() to generate the final CRB.

        Parameters
        ----------
        constraints : np.ndarray, optional
            Constraints matrix for the eventual CRB. This should be provided
            as a row-wise matrix; one row for each constraint.
        """

        self.components = []
        self.constraints = constraints
        if self.constraints is not None:
            if self.constraints.ndim == 1:
                self.constraints = self.constraints.reshape((1, -1)) # make it into a row

    def addComponent(self, component: LocalizationCRBComponent):
        """
        Add a LocalizationCRBComponent subclass.

        Parameters
        ----------
        component : LocalizationCRBComponent subclass instance
            This component should correspond to a measurement.

        Returns
        -------
        self
            Returns the current instance so that addComponent() calls
            can be chained like .addComponent().addComponent().
        """
        self.components.append(component)
        return self

    def fim(self) -> np.ndarray:
        """
        Calculate the Fisher Information Matrix.
        Usually you can call compute() to generate the final CRB directly.

        Returns
        -------
        np.ndarray
            Fisher information matrix.
        """
        # Calculate FIMs for each component
        fim_mat = np.zeros((3,3), np.float64)
        for component in self.components:
            fim_mat += component.fim()

        return fim_mat

    def compute(self) -> np.ndarray:
        """
        Computes the CRB based on all components.

        Returns
        -------
        np.ndarray
            Final CRB as a square matrix, including constraints if provided.
        """
        fim = self.fim()

        # Compute FIM 
        if self.constraints is not None:
            U = sp.linalg.null_space(self.constraints)
            crb = U @ np.linalg.inv(U.T @ fim @ U) @ U.T
        else:
            crb = np.linalg.inv(fim)

        return crb


#%%
if __name__ == "__main__":
    # from localizationRoutines import *
    # x = np.zeros(3)
    # S = np.array([
    #     [1, 0, 0],
    #     [-1, 0, 0],
    #     [0, 1, 0],
    #     [0, -1, 0]
    # ]).T

    # sig_r = 1e-6
    # crb_old, fim_old = calcCRB_TD(x, S, np.ones(2)*sig_r, 
    #                               pairs=np.array([[1,0],[3,2]]),
    #                               cmat=np.array([0,0,1]).reshape((-1,1)))
    # print(fim_old)
    # print(crb_old)
    # print("-------------------------")

    # sig_td = sig_r / speed_of_light
    # inv_sigma_td_sq = 1/sig_td**2
    # comp1 = TDOACRBComponent(x, inv_sigma_td_sq, S.T[[0,1],:])
    # print(comp1.r)
    # comp2 = TDOACRBComponent(x, inv_sigma_td_sq, S.T[[2,3],:])
    # print(comp2.r)
    # crb = CRB(constraints=np.array([0,0,1]))
    # crb.addComponent(comp1).addComponent(comp2)#.addComponent(comp3)
    # print(fim)
    # print(crb.compute())

    # print("==========================")
    # print(crb_old - crb.compute())
    # print(crb_old / crb.compute())

    # print(crb_old @ fim_old)
    # print(crb.compute() @ fim)

    # assert(False)

    # Begin unittests
    import unittest
    from scipy.spatial.transform import Rotation as R

    class TestBasicCRBComponent(unittest.TestCase):
        def test_position_shape_throw(self):
            # Must throw ValueError if x is not a (3, ) array
            self.assertRaises(
                ValueError,
                LocalizationCRBComponent,
                np.zeros(3).reshape((-1,1)),
                1.0,
                np.ones(3) 
            )
        def test_unimplemented_base_differentiate(self):
            # Create the non subclass one and show that it throws
            self.assertRaises(
                NotImplementedError,
                LocalizationCRBComponent,
                np.zeros(3),
                1.0,
                np.ones(3) 
            )

        def test_sigma_type_error(self):
            # Must throw type error if sigma is not float/np.ndarray
            self.assertRaises(
                TypeError,
                LocalizationCRBComponent,
                np.zeros(3),
                1,
                np.ones(3) 
            )

    class TestBasicCRB(unittest.TestCase):
        def test_instantiation(self):
            # Show that constraints are automatically made into 2d
            constraint = np.ones(3)
            crb = CRB(constraint)
            self.assertEqual(crb.constraints.shape, (1, 3))


        def test_chained_adds(self):
            # Show that addComponent() is able to chain calls
            crb = CRB()
            comp1 = AOA3DCRBComponent(np.zeros(3), 1.0, np.ones(3))
            comp2 = AOA3DCRBComponent(np.ones(3), 1.0, 2*np.ones(3))
            crb.addComponent(comp1).addComponent(comp2)

    class TestTDOACRBComponent(unittest.TestCase):
        def setUp(self):
            self.x = np.zeros(3)
            self.S = np.array([
                [1, 0, 0], # Right
                [-1, 0, 0], # Left
                [0, 1, 0], # Forward
                [0, -1, 0], # Backward
                [0, 0, 1], # Top
                [0, 0, -1] # Bottom
            ])

            self.sig_r = 1e-6
            self.sig_td = self.sig_r / speed_of_light
            self.inv_sigma_td_sq = 1/self.sig_td**2

        def test_plane_constrained_tdoa(self):
            # Target and sensors all in the plane in an equidistant simple diamond
            # We expect the CRB error variance components to be exactly equal to the originally input errors
            comp1 = TDOACRBComponent(self.x, self.inv_sigma_td_sq, self.S[[0,1],:]) # Use left/right
            comp2 = TDOACRBComponent(self.x, self.inv_sigma_td_sq, self.S[[2,3],:]) # use forward/backward
            crb = CRB(constraints=np.array([0,0,1])) # Constrain to the plane
            crb.addComponent(comp1).addComponent(comp2)

            # Should be diagonal
            self.assertEqual(
                crb.compute()[0,0], crb.compute()[1,1]
            )
            # Z component should be 0 since constrained to plane
            self.assertEqual(
                crb.compute()[2,2], 0
            )
            # Non-zero components should be equal to the original sig_r squared
            self.assertAlmostEqual(
                crb.compute()[0,0],
                self.sig_r**2
            )
            self.assertAlmostEqual(
                crb.compute()[1,1],
                self.sig_r**2
            )

        def test_3d_unconstrained_tdoa(self):
            # We now have all 6 sensors in a 3d equidistant diamond/cube
            # Similarly, CRB error variance components should be exactly equal to the originally input errors
            comp1 = TDOACRBComponent(self.x, self.inv_sigma_td_sq, self.S[[0,1],:]) # Use left/right
            comp2 = TDOACRBComponent(self.x, self.inv_sigma_td_sq, self.S[[2,3],:]) # use forward/backward
            comp3 = TDOACRBComponent(self.x, self.inv_sigma_td_sq, self.S[[4,5],:]) # use top/bottom
            crb = CRB()
            crb.addComponent(comp1).addComponent(comp2).addComponent(comp3)

            # Should be diagonal
            self.assertEqual(
                crb.compute()[0,0], crb.compute()[1,1]
            )
            self.assertEqual(
                crb.compute()[0,0], crb.compute()[2,2] 
            )
            # Non-zero components should be equal to the original sig_r squared
            self.assertAlmostEqual(
                crb.compute()[0,0],
                self.sig_r**2
            )
            self.assertAlmostEqual(
                crb.compute()[1,1],
                self.sig_r**2
            )
            self.assertAlmostEqual(
                crb.compute()[2,2],
                self.sig_r**2
            )


    class TestAOA3DCRBComponent(unittest.TestCase):
        def expectedTrace(self, delta: float, length: float):
            return np.tan(delta) * length

        def setUp(self):
            # Create the base x-axis aligned vectors
            self.x = np.array([1,0,0], np.float64)
            self.S = np.zeros(3, np.float64)
            self.delta = np.random.rand() * 0.01 # Use randomised delta from 0 to 0.01 radians (about 0.573 deg)
            # DEV NOTE: using delta = 0.1 means that you must raise the rtol to 0.01 (1%);
            # This corresponds to how we are making many small angle approximations and dropping higher order terms
            # Hence the approximation becomes worse as the error becomes bigger.
        
            # Create some arbitrary random rotations 
            self.numRots = 100 # Number of randomised rotations to test
            phi_rots = np.random.rand(self.numRots) * 2 * np.pi
            theta_rots = np.random.rand(self.numRots) * np.pi - np.pi/2
            # Twist around z-axis to get phi
            phi_rotmats = [R.from_rotvec([0,0,phi_rot]).as_matrix() for phi_rot in phi_rots] 
            # Twist around y-axis to get theta
            theta_rotmats = [R.from_rotvec([0,theta_rot,0]).as_matrix() for theta_rot in theta_rots]
            self.rotmats = [
                phi_rotmats[i] @ theta_rotmats[i] for i in range(self.numRots)
            ] # We do theta rotation first to achieve the actual theta from pre-defined x-vector

            self.rtol = 0.001 # What we expect our approximation to be equal to
            print("Delta = %f rad (%f deg), Using relative tolerance of %f" % (
                self.delta, self.delta/np.pi*180.0, self.rtol))
           
        def test_3d_rotations(self):
            for rotmat in self.rotmats:
                rotated_x = rotmat @ self.x
                aoacrb = AOA3DCRBComponent(rotated_x, self.delta, self.S)
                # Check simple things first
                self.assertTrue(np.allclose(rotated_x - self.S, aoacrb.uf))
                # Create the CRB
                crb = CRB(aoacrb.u)
                crb.addComponent(aoacrb)
                self.assertTrue(
                    np.allclose(
                        np.trace(crb.compute())**0.5, 
                        self.expectedTrace(self.delta, np.linalg.norm(aoacrb.uf)), rtol=self.rtol
                    )
                )
        
        def test_scaling_alone(self):
            for i in range(self.numRots):
                scale = np.random.rand() * 100000
                scaled_x = scale * self.x
                aoacrb = AOA3DCRBComponent(scaled_x, self.delta, self.S)
                # Check simple things first
                self.assertTrue(np.allclose(scaled_x - self.S, aoacrb.uf))
                # Create the CRB
                crb = CRB(aoacrb.u)
                crb.addComponent(aoacrb)
                self.assertTrue(
                    np.allclose(
                        np.trace(crb.compute())**0.5, 
                        self.expectedTrace(self.delta, np.linalg.norm(aoacrb.uf)), rtol=self.rtol
                    )
                )

        def test_scale_and_rotate(self):
            for rotmat in self.rotmats:
                scale = np.random.rand() * 100000
                new_x = scale * rotmat @ self.x
                aoacrb = AOA3DCRBComponent(new_x, self.delta, self.S)
                # Check simple things first
                self.assertTrue(np.allclose(new_x - self.S, aoacrb.uf))
                # Create the CRB
                crb = CRB(aoacrb.u)
                crb.addComponent(aoacrb)
                self.assertTrue(
                    np.allclose(
                        np.trace(crb.compute())**0.5, 
                        self.expectedTrace(self.delta, np.linalg.norm(aoacrb.uf)), rtol=self.rtol
                    )
                )


    class TestTOACRBComponent(unittest.TestCase):
        def test_simple(self):
            # We create a plane and two TOAs, equidistant
            x = np.array([0,100,0], np.float64)
            inv_sigma_tau_sq = 1/(1e-10)**2
            s1 = np.array([100,0,100], np.float64)
            s2 = np.array([-100,0,100], np.float64)
            

            # Construct CRB with x-y plane constraint
            crb = CRB(constraints=np.array([0,0,1]))
            crb.addComponent(
                TOACRBComponent(
                    x, inv_sigma_tau_sq, s1
                )
            ).addComponent(
                TOACRBComponent(
                    x, inv_sigma_tau_sq, s2
                )
            )

            # Compute
            crbmat = crb.compute()
            u, s, vh = np.linalg.svd(crbmat)

            # We expect the two eigenvalues to be equal
            self.assertAlmostEqual(
                s[0], s[1]
            )
            # The two eigenvectors can be in any direction at this point so it's irrelevant
            # i.e. perfect circle on the plane

                

    # Run tests
    unittest.main(verbosity=2)



