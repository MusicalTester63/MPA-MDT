# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 15:45:22 2020

@author: Dell
"""
from random import random
import astropy.units as u
import numpy as np
from numpy import cos, sin, tan, pi,degrees,array
from astropy.coordinates import EarthLocation
# phi = EarthLocation(lat=0*u.deg,lon=90*u.deg,
#                          height=0*u.m).lat.rad

#%%

def generate_params(mag):
    params={}
    names = ['zh','zd','co','fo','mp','ma','me','tf','df']

    for j in names:
        if j=='zh': 
            params.update({j:random()*mag*10})
        elif j=='zd':
            params.update({j:random()*mag*10})
        else:
            params.update({j:random()*mag})
    return params



def point_ha2(t,dec,phi,zh,co,mp,ma,me,tf,df):
    tc=zh+co/(cos(dec))+mp*tan(dec)-ma*cos(t)*tan(dec)+\
       me*sin(t)*tan(dec) + tf*cos(phi)*sin(t)/cos(dec) -\
       df*(cos(phi)*cos(t)+sin(phi)*tan(dec))
    return tc

def point_dec2(dec,t,phi,zd,fo,ma,me,tf):
    dc=zd+ma*sin(t)+me*cos(t) + tf*(cos(phi)*cos(t)*sin(dec)-\
       sin(phi)*cos(dec)) + fo*cos(t)
    return dc

def point(params, t, dec, phi):
    a = point_ha2(t,dec,phi,params['zh'],params['co'],params['mp'],
                        params['ma'], params['me'], params['tf'], params['df'])
    b = point_dec2(dec, t, phi, params['zd'], params['fo'], 
                            params['ma'], params['me'], params['tf'])
    return array([a,b])

