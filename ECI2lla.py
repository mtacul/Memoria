# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 22:31:24 2023

@author: nachi
"""
from datetime import datetime
from skyfield.api import Distance, load, wgs84, utc
from skyfield.positionlib import Geocentric
import numpy as np

#%%eci a lla

def eci2lla(posicion, fecha):
    ts = load.timescale()
    fecha = fecha.replace(tzinfo=utc)
    t = ts.utc(fecha)
    d = [Distance(m=i).au for i in (posicion[0]*1000, posicion[1]*1000, posicion[2]*1000)]
    p = Geocentric(d,t=t)
    g = wgs84.subpoint(p)
    latitud = g.latitude.degrees
    longitud = g.longitude.degrees
    altitud = g.elevation.m
    return latitud, longitud, altitud
#%% prueba funcion
start_time_gps0 = datetime(2023, 7, 5, 12, 0, 0,tzinfo=utc)  # Ejemplo: 5 de julio de 2023, 12:00:00
posicion = np.array([885.8137390481352, 2776.8568064255096, -6212.451180356124])
lla = eci2lla(posicion,start_time_gps0)
print(lla)

#%%
ts = load.timescale()
start_time_gps = datetime(2023, 7, 5, 12, 0, 0,tzinfo=utc)  # Ejemplo: 5 de julio de 2023, 12:00:00
start_time_gps = start_time_gps.replace(tzinfo=utc)
t = ts.utc(start_time_gps)
#t = ts.utc(2020, 11, 6)

d = [Distance(m=i).au for i in (885.8137390481352*1000, 2776.8568064255096*1000, -6212.451180356124*1000)]
p = Geocentric(d, t=t)
g = wgs84.subpoint(p)
print(g.latitude.degrees, 'degrees latitude')
print(g.longitude.degrees, 'degrees longitude')
print(g.elevation.m, 'meters above WGS84 mean sea level')