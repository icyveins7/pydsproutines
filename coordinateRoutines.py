import numpy as np

from skyfield.api import wgs84, load, Distance
from skyfield.toposlib import ITRSPosition


def geodeticLLA2ecef(lat_rad, lon_rad, h, checkRanges=False):
    # Some error checking
    if checkRanges and (
        np.any(np.abs(lat_rad) > np.pi / 2) or np.any(np.abs(lon_rad) > np.pi)
    ):
        raise ValueError(
            "Latitude and longitude magnitudes are too large. Did you forget to convert to radians?"
        )
    # This should replicate wgs84.latlon().itrs_xyz.m (which also works on arrays)
    # Speedwise, this is about 2-3x faster, since it skips all the object instantiations
    # Reference https://en.wikipedia.org/wiki/Geodetic_coordinates
    a = 6378137.0
    # WGS84 constants, reference https://en.wikipedia.org/wiki/World_Geodetic_System
    b = 6356752.314245
    N = a**2 / np.sqrt(a**2 * np.cos(lat_rad) ** 2 +
                       b**2 * np.sin(lat_rad) ** 2)

    x = (N + h) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + h) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (b**2 / a**2 * N + h) * np.sin(lat_rad)

    return np.vstack((x, y, z))


def ecef2geodeticLLA(x: np.ndarray):
    if not isinstance(x, np.ndarray):
        raise TypeError("Must be numpy array.")

    now = load.timescale().now()  # Can re-use since ITRS is time-agnostic

    # A single 3-d position
    if x.ndim == 1 and x.size == 3:
        x = x.reshape((1, 3))

    # Multiple 3-d positions, one in each column
    if x.shape[0] == 3:
        pos = ITRSPosition(Distance(m=x))  # This accepts a 3xN array directly
        latlonele = wgs84.geographic_position_of(pos.at(now))
        lle_arr = np.vstack(
            (
                latlonele.latitude.degrees,
                latlonele.longitude.degrees,
                latlonele.elevation.m,
            )
        )
        return lle_arr

    else:
        raise ValueError("Invalid dimensions; expected 3xN.")
