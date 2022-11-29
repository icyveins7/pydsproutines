# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 13:21:13 2022

@author: lken
"""

import numpy as np

from sgp4.api import Satrec, SatrecArray, jday
 # Apparently, generating satellite positions with WGS72 is more accurate, as that is what the TLEs are generated from
from sgp4.api import WGS72OLD, WGS72, WGS84

#%% Some convenient wrappers



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
    