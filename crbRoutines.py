import numpy as np
import scipy as sp


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

class TDOAComponent(LocalizationCRBComponent):
    def __init__(
        self,
        x: np.ndarray,
        inv_sigma_td_sq: float,
        S: np.ndarray
    ):
        super().__init__(x, inv_sigma_td_sq, S)
        self.r = np.linalg.norm(x - S)

    def _differentiate(self):
        r_dx = (self.x - self.S) / self.r
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

    def compute(self) -> np.ndarray:
        """
        Computes the CRB based on all components.

        Returns
        -------
        np.ndarray
            Final CRB as a square matrix, including constraints if provided.
        """
        # Calculate FIMs for each component
        fim = np.zeros((3,3), np.float64)
        for component in self.components:
            fim += component.fim()

        # Compute FIM 
        if self.constraints is not None:
            U = sp.linalg.null_space(self.constraints)
            crb = U @ np.linalg.inv(U.T @ fim @ U) @ U.T
        else:
            crb = np.linalg.inv(fim)

        return crb


#%%
if __name__ == "__main__":
    # Create the AOA3DCRBComponent for simple case along x-axis
    # x = np.array([100,0,0])
    # x = np.array([0,100,0], np.float64)
    # angle = np.random.rand() * np.pi - np.pi/2 # Between -pi/2 and pi/2
    # x = np.array([100*np.cos(angle), 0, 100*np.sin(angle)], np.float64)
    # S = np.zeros(3)
    # delta = 0.1
    # aoacrb = AOA3DCRBComponent(x, delta, S)
    # print(aoacrb.inv_sigma_sq)
    # print(aoacrb.uf)
    # print(aoacrb.u)
    # print(aoacrb.dphi)
    # print(aoacrb.dtheta)
    # print(aoacrb.partials)
    # print(aoacrb.fim())


    # # Create the CRB for this and add the component
    # planeconstraint = aoacrb.u # We use the direction vector as the normal
    # crb = CRB(planeconstraint)
    # crb.addComponent(aoacrb)
    # print(crb.constraints)
    # print(crb.compute())
    # print(np.trace(crb.compute()))


    # Begin unittests
    import unittest
    from scipy.spatial.transform import Rotation as R

    class TestBasicCRBComponent(unittest.TestCase):
        def test_basic(self):
            # Create the non subclass one and show that it throws
            self.assertRaises(
                NotImplementedError,
                LocalizationCRBComponent,
                np.zeros(3),
                1.0,
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


                

    # Run tests
    unittest.main(verbosity=2)



