# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 13:21:13 2022

@author: lken
"""

import numpy as np
import re

from sgp4.api import Satrec, SatrecArray, jday
 # Apparently, generating satellite positions with WGS72 is more accurate, as that is what the TLEs are generated from
from sgp4.api import WGS72OLD, WGS72, WGS84
from sgp4.api import SGP4_ERRORS

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


#%% Testing
if __name__ == "__main__":
    s = '1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991'
    t = '2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482'
    satellite = Satrec.twoline2rv(s, t, WGS72)
    
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
    