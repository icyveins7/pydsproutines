# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 13:21:13 2022

@author: lken
"""

import numpy as np
import re
import datetime as dt

from sgp4.api import Satrec, SatrecArray, jday

# Apparently, generating satellite positions with WGS72 is more accurate, as that is what the TLEs are generated from
from sgp4.api import WGS72OLD, WGS72, WGS84
from sgp4.api import SGP4_ERRORS
from skyfield.positionlib import Geocentric

# %% Wrapper over skyfield's EarthSatellite, to restore constants functionality
from skyfield.api import EarthSatellite, load, wgs84
from skyfield.framelib import itrs

from localizationRoutines import geodeticLLA2ecef

import matplotlib.pyplot as plt


class Satellite(EarthSatellite):
    """
    Refer to the source code on github to confirm that these are the only changes necessary.

    This can then be used as a drop in replacement for skyfield.
    """

    def __init__(self, line1, line2, name=None, ts=None, const=WGS72):
        super().__init__(line1, line2, name=name, ts=ts)  # This ignores the const
        # So remake the satrec with the const now
        self.model = Satrec.twoline2rv(line1, line2, const)
        self._setup(self.model)

    def quickplot(self, gpstime: np.ndarray, rxposlla: np.ndarray):
        gc = sf_propagate_satellite_to_gpstime(self, t)
        lla = wgs84.geographic_position_of(gc)
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(lla.longitude.degrees,
                   lla.latitude.degrees)
        # TODO: check below code for shapes
        rxposecef = geodeticLLA2ecef(rxposlla[0], rxposlla[1], rxposlla[2])
        rxecef = sf_geocentric_to_itrs(gc)
        connector = rxecef - rxposecef
        normal = rxecef / np.array(
            [wgs84.radius.m, wgs84.radius.m, wgs84.polar_radius.m]
        )**2

        theta = np.arccos(
            np.dot(normal, connector) /
            np.linalg.norm(normal) /
            np.linalg.norm(connector, axis=1)
        )

        thetadeg = np.rad2deg(np.pi/2 - theta)

        ax[1].plot(thetadeg)

        return fig, ax


# %% Below we list some common functionality in wrapped functions
# They are more of a quick reference so we don't have to look it up in the docs.


def sf_propagate_satellite_to_gpstime(satellite: Satellite, gpstime: float):
    """
    Propagate a satellite to a GPSTime.
    Note that this is not a very optimized function.

    Parameters
    ----------
    satellite : EarthSatellite
        A satellite class instance.
    gpstime : float or list or np.ndarray
        A GPS time or iterable of times that is locked to UTC.

    Returns
    -------
    pv : Geocentric
        Class instance of the satellite's position/velocity vector.
        You probably want to turn this into another frame.
    """
    ts = load.timescale()
    if isinstance(gpstime, float):
        dd = [dt.datetime.fromtimestamp(gpstime, tz=dt.timezone.utc)]
    elif hasattr(gpstime, "__iter__") and not isinstance(gpstime, str):
        dd = [dt.datetime.fromtimestamp(
            i, tz=dt.timezone.utc) for i in gpstime]
    else:
        raise TypeError("gpstime must be float or iterable")
    t = ts.from_datetimes(
        dd
    )  # Array container, this lets you avoid multiple .at() calls
    return satellite.at(t)


def sf_geocentric_to_itrs(geocentric: Geocentric, returnVelocity: bool = False):
    """
    Convert a geocentric position/velocity to ITRS.

    Parameters
    ----------
    geocentric : Geocentric
        A geocentric frame class instance. This is the natural result from
        sf_propagate_satellite_to_gpstime().
    returnVelocity : bool, optional
        If True, return the velocity vector as well. The default is False.

    Returns
    -------
    itrs : ITRS
        Position/velocity vector in ITRS.
        This is effectively the ECEF frame.
    """
    if returnVelocity:
        return geocentric.frame_xyz_and_velocity(itrs)
    else:
        return geocentric.frame_xyz(itrs)


# %% Testing
if __name__ == "__main__":
    s = "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991"
    t = "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482"
    satellite = Satrec.twoline2rv(s, t, WGS72)
    print(satellite.jdsatepoch)
    print(satellite.jdsatepochF)

    jd, fr = jday(2019, 12, 9, 12, 0, 0)
    e, r, v = satellite.sgp4(jd, fr)
    print(e)
    print(r)
    print(v)

    jd = np.array((2458826, 2458826, 2458826, 2458826))
    fr = np.array((0.0001, 0.0002, 0.0003, 0.0004))
    e, r, v = satellite.sgp4_array(jd, fr)
    print(e)
    print(r)
    print(v)

    satellites = SatrecArray([satellite, satellite])
    e, r, v = satellites.sgp4(jd, fr)
    print(e)
    print(r)
    print(v)

    # Create Satellite object
    print("===================================")
    from timingRoutines import Timer

    timer = Timer()

    sat = Satellite(s, t, const=WGS84)
    t = [i + 1575849600.0 for i in range(0, 86400)]
    satgc = sf_propagate_satellite_to_gpstime(sat, t)

    # Go to LLA using skyfield
    timer.start()
    satgc_geog = wgs84.geographic_position_of(satgc)
    satgc_lla = np.vstack(
        (
            satgc_geog.latitude.degrees,
            satgc_geog.longitude.degrees,
            satgc_geog.elevation.m,
        )
    )
    timer.end("direct to geodetic LLA")
    print(satgc_lla)

    # Go to LLA using ITRS first, then custom conversion
    from localizationRoutines import *

    timer.start()
    satecef = sf_geocentric_to_itrs(satgc)
    timer.evt("to ITRS")
    # print(satecef.m)
    satecef_lla = ecef2geodeticLLA(satecef.m)
    timer.end("then to geodetic LLA")
    print(satecef_lla)
