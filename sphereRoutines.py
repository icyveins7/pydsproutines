# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 15:40:43 2022

@author: Lken
"""

import numpy as np
import pyqtgraph.opengl as gl

from plotRoutines import *


# %%
class Ellipsoid:
    def __init__(
        self,
        a: float,
        b: float,
        c: float,
        mu: np.ndarray = np.zeros(3),
        Rx: np.ndarray = np.eye(3),
        Rz: np.ndarray = np.eye(3),
    ):
        """
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

        """

        self.a = a
        self.b = b
        self.c = c
        self.mu = mu
        self.Rx = Rx
        self.Rz = Rz

    def pointsFromAngles(self, theta, phi):
        points = np.array(
            [
                self.a * np.sin(theta) * np.cos(phi),
                self.b * np.sin(theta) * np.sin(phi),
                self.c * np.cos(theta),
            ]
        )
        return points

    def transform(self, points):
        if points.ndim == 3:
            return points + self.mu.reshape((-1, 1, 1))
        else:
            return points + self.mu.reshape((-1, 1))

    # Other methods
    def visualize(
        self,
        theta=np.arange(0, np.pi, 0.001),
        phi=np.arange(-np.pi, np.pi, 0.1),
        ax=None,
        colour="k",
    ):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
        else:
            fig = None

        # Generate some points in a mesh
        theta, phi = np.meshgrid(theta, phi)

        # Calculate the cartesian coordinates from parametrisation
        points = self.pointsFromAngles(theta, phi)
        # breakpoint()
        points = self.transform(points)

        # Ensure equal ratios
        ax.set_box_aspect(
            (
                np.ptp(points[0].reshape(-1)),
                np.ptp(points[1].reshape(-1)),
                np.ptp(points[2].reshape(-1)),
            )
        )

        ax.plot_wireframe(points[0], points[1], points[2], color=colour)

        return ax, fig

    def intersectRay(self, s: np.ndarray, direction: np.ndarray):
        # Check 1-D inputs
        if s.ndim != 1 or direction.ndim != 1:
            raise ValueError("s and direction must be 1-D arrays")

        # Calculate the coefficients of the quadratic equation, ax^2 + bx + c = 0
        denominatorsq = np.array([self.a**2, self.b**2, self.c**2])
        sprime = s - self.mu  # Need to offset by the ellipsoid centre
        coeffs = np.array(
            [
                np.sum(sprime**2 / denominatorsq) - 1.0,  # c
                np.sum(2 * sprime * direction / denominatorsq),  # b
                np.sum(direction**2 / denominatorsq),  # a
            ]
        )
        poly = np.polynomial.Polynomial(coeffs)

        # Find the real positive roots
        lmbda = poly.roots()
        lmbda = lmbda[np.isreal(lmbda)]
        lmbda = lmbda[lmbda >= 0]
        if lmbda.size > 0:
            # Pick the smallest positive one
            lmbda = np.min(lmbda)
            # Propagate the ray to the point
            x = s + direction * lmbda
            return x

        else:
            return None


class OblateSpheroid(Ellipsoid):
    def __init__(
        self,
        omega: float,
        lmbda: float,
        mu: np.ndarray = np.zeros(3),
        Rx: np.ndarray = np.eye(3),
        Rz: np.ndarray = np.eye(3),
    ):

        self.omega = omega
        self.lmbda = lmbda
        assert lmbda < omega
        super().__init__(omega, omega, lmbda, mu, Rx, Rz)


class Sphere(Ellipsoid):
    def __init__(self, r: float, mu: np.ndarray = np.zeros(3)):
        self.r = r
        super().__init__(r, r, r, mu)

    def intersectOblateSpheroid(self, theta, omega, lmbda):
        rs = self.r * np.sin(theta)
        rc = self.r * np.cos(theta)

        # Compute expanded coefficients
        gamma = lmbda**2 * (rs**2 + self.mu[0] ** 2 + self.mu[1] ** 2)
        beta = omega**2 * (rc**2 + 2 * rc * self.mu[2] + self.mu[2] ** 2)
        A = lmbda**2 * 2 * rs * self.mu[0]
        B = lmbda**2 * 2 * rs * self.mu[1]

        with np.errstate(divide="ignore", invalid="ignore"):
            alpha = np.arctan2(B, A)  # do not use arctan! make sure its arctan2!
            t = (lmbda**2 * omega**2 - beta - gamma) / np.sqrt(A**2 + B**2)
            # breakpoint()
            basic = np.arccos(t)  # returns [0, pi]

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


# %% Testing
if __name__ == "__main__":
    closeAllFigs()
    sphere = Sphere(1, np.array([-1, -2, 0]))

    theta = np.arange(0, np.pi, 0.01)
    omega = 2.0
    lmbda = 1.5
    points = sphere.intersectOblateSpheroid(theta, omega, lmbda)

    # Show the sphere
    ax, fig = sphere.visualize()

    # Show the oblatespheroid
    oblate = OblateSpheroid(omega, lmbda)
    oblate.visualize(ax=ax, colour="b")
    # # Create spheroid
    ax.plot3D(points[0], points[1], points[2], "r-")

    # Check if points truly lie on surface
    check = (
        points[0, :] ** 2 / omega**2
        + points[1, :] ** 2 / omega**2
        + points[2, :] ** 2 / lmbda**2
    )
    print(check)

    # Create a ray and an oblate spheroid
    s = np.array([3, 3, 3])
    d = np.array([-1, -1.1, -1.2])
    # Find intersection
    intersection = oblate.intersectRay(s, d)
    # breakpoint()
    if intersection is not None:
        # Check if points truly lie on surface
        check = (
            intersection[0] ** 2 / omega**2
            + intersection[1] ** 2 / omega**2
            + intersection[2] ** 2 / lmbda**2
        )
        print(check)
        # Check if the direction vector is back-calculated correctly
        check = intersection - s
        print(check / np.linalg.norm(check), d / np.linalg.norm(d))

        ax2, fig2 = oblate.visualize(
            np.arange(0, np.pi, 0.05), np.arange(0, np.pi, 0.05)
        )
        ax2.plot3D(
            [s[0], intersection[0]],
            [s[1], intersection[1]],
            [s[2], intersection[2]],
            "rx-",
        )

    plt.show()
