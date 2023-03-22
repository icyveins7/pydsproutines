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


#%% TLE Parser
class TLEfile:
    def __init__(self, filepath: str):
        self.filepath = filepath # To help remember which file this is
        with open(filepath, "r") as fid:
            self.lines = fid.readlines()
            
        self.tles = self._parse()
            
    # Internal parsing method
    def _parse(self):
        tles = dict()
        currentSat = None
        for line in self.lines:
            if not re.match("\\d \\d+", line):
                currentSat = line.strip()
                tles[currentSat] = []
            else:
                tles[currentSat].append(line.strip())
                assert(len(tles[currentSat]) <= 2)
                
        return tles
    
    # Convenient getter
    def __getitem__(self, key):
        return self.tles[key]
        

#%% Some convenient wrappers for pure SGP4
class Satellite:
    # Redirector for constants
    consts = {
        'wgs84': WGS84,
        'wgs72': WGS72,
        'wgs72old': WGS72OLD
    }
    
    def __init__(self, tle: list, name: str=None, const: str='wgs84'):
        self.tle = tle
        self.name = name
        self.const = self.consts[const]
        # Call the sgp4 stuff
        self.satrec = Satrec.twoline2rv(*tle, self.const)
        
    @classmethod
    def fromName(cls, tle: TLEfile, name: str, const: str='wgs84'):
        '''
        Automatically constructs from a TLEfile instance (see above) and a desired satellite name.

        Parameters
        ----------
        tle : TLEfile
            A prepared TLEfile class instance.
        name : str
            Desired name of satellite.
        const : str, optional
            Gravity constant. The default is 'wgs84'.

        '''
        return cls(tle[name], name, const)
    
    def _parseErrors(self, e: int):
        if np.any(e != 0):
            # Get the first instance and return the error
            idx = np.argmax(e != 0) # This returns on the first True
            raise Exception(SGP4_ERRORS[e[idx]])
    
    ### Main propagation methods
    def propagate(self, jd: np.ndarray, fr: np.ndarray):
        '''
        This is just a redirect to the sgp4 call. We assume array inputs.
        '''
        e, r, v = self.satrec.sgp4_array(jd, fr)
        self._parseErrors(e)
        return r, v
    
    def propagateFrom(self, jd0: float, fr0: float, start: float, stop: float, step: float):
        fr = fr0 + np.arange(start, stop, step)
        jd = jd0 + np.zeros(fr.size)
        r, v = self.propagate(jd, fr)
        return r, v
    
    def propagateFromEpoch(self, start: float, stop: float, step: float):
        r, v = self.propagateFrom(
            self.satrec.jdsatepoch,
            self.satrec.jdsatepochF,
            start,
            stop,
            step
        )
        return r, v
    
    def propagateSecondsFromEpoch(self, seconds: float):
        '''
        Not advisable to use this as it reverts to Pythonic scalars, hence slow.
        '''
        e, r, v = self.satrec.sgp4_tsince(seconds/60.0)
        return e, r, v
        
    def propagateUtcTimestamps(self, timestamps: np.ndarray):
        pass

#%% However, skyfield offers very similar functionality
from skyfield.api import EarthSatellite, load, wgs84
from skyfield.framelib import itrs

#%% Below we list some common functionality in wrapped functions
# They are more of a quick reference so we don't have to look it up in the docs.

def sf_get_satellite_from_tle(tle: TLEfile, name: str):
    """
    Get a satellite from a TLE file.

    Parameters
    ----------
    tle : TLEfile
        A path to a TLE file.
    name : str
        Desired name of satellite.

    Returns
    -------
    satellite : Satellite
        A satellite class instance.
    """
    satellites = load.tle_file(tle)
    # Load by name
    satellites = {sat.name: sat for sat in satellites}
    return satellites[name]

def sf_propagate_satellite_to_gpstime(satellite: EarthSatellite, gpstime: float):
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
    t = ts.utc(dd.year, dd.month, dd.day, dd.hour, dd.minute, dd.second)
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
    
    #%%
    sat = Satellite([s,t])
    r, v = sat.propagate(jd, fr)
    r, v = sat.propagateFromEpoch(-1.0, 1.0, 0.001)
    