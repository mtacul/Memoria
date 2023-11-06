# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 17:04:22 2023

@author: nachi
"""

#!python setup.py install
import pyIGRF

# Definir la ubicación geográfica (latitud, longitud) en grados decimales
lat = 40.0
lon = -75.0

# Definir la altitud en metros sobre el nivel del mar
alt = 0.0

# Definir la fecha para la cual deseas calcular el campo geomagnético
fecha = 2023

campo_geomagnetico = pyIGRF.igrf_value(lat, lon, alt, fecha)

print("Campo geomagnético (nT):")
print("Componente X:", campo_geomagnetico[3])
print("Componente Y:", campo_geomagnetico[4])
print("Componente Z:", campo_geomagnetico[5])
