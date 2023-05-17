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

#%% Wrapper over skyfield's EarthSatellite, to restore constants functionality
from skyfield.api import EarthSatellite, load, wgs84
from skyfield.framelib import itrs

class Satellite(EarthSatellite):
    """
    Refer to the source code on github to confirm that these are the only changes necessary.
    
    This can then be used as a drop in replacement for skyfield.
    """
    def __init__(self, line1, line2, name=None, ts=None, const=WGS72):
        super().__init__(line1, line2, name=name, ts=ts) # This ignores the const
        # So remake the satrec with the const now
        self.model = Satrec.twoline2rv(line1, line2, const)
        self._setup(self.model)

#%% Below we list some common functionality in wrapped functions
# They are more of a quick reference so we don't have to look it up in the docs.

def sf_propagate_satellite_to_gpstime(satellite: Satellite, gpstime: float):
    """
    Propagate a satellite to a GPSTime.
    Note that this is not a very optimized function.

    Parameters
    ----------
    satellite : EarthSatellite
        A satellite class instance.
    gpstime : float
        A GPS time that is locked to UTC.

    Returns
    -------
    pv : Geocentric
        Class instance of the satellite's position/velocity vector.
        You probably want to turn this into another frame.
    """
    ts = load.timescale()
    dd = dt.datetime.fromtimestamp(gpstime, tz=dt.timezone.utc)
    t = ts.utc(dd.year, dd.month, dd.day, dd.hour, dd.minute, dd.second+dd.microsecond/1e6)
    return satellite.at(t)

def sf_geocentric_to_itrs(geocentric: Geocentric, returnVelocity: bool=False):
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



#%% Testing
if __name__ == "__main__":
    s = '1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991'
    t = '2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482'
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
    