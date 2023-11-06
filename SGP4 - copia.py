# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 04:05:44 2023

@author: nachi
"""

#%%
from pyproj import Proj
from datetime import datetime, timedelta
from sgp4.earth_gravity import wgs84
from sgp4.io import twoline2rv
import numpy as np

#%%

# Define la ubicación de tu archivo TLE
tle_file = "suchai_3.txt"

# Lee el contenido del archivo
with open(tle_file, "r") as file:
    tle_lines = file.read().splitlines()

# Asegúrate de que el archivo contiene al menos dos líneas de TLE
if len(tle_lines) < 2:
    print("El archivo TLE debe contener al menos dos líneas.")
else:
    # Convierte las líneas del archivo en un objeto Satrec
    satellite = twoline2rv(tle_lines[0], tle_lines[1], wgs84)

    # Define la fecha inicial 
    start_time = datetime(2023,7,5, 12, 0, 0)  # Ejemplo: 5 de julio de 2023, 12:00:00
    print(start_time)
    # Define el tiempo de propagación en segundos
    propagation_time = 1440

    # Calcula la fecha final
    end_time = start_time + timedelta(seconds=propagation_time)

    # Propaga el satélite durante el tiempo especificado
    position, velocity = satellite.propagate(
        start_time.year, start_time.month, start_time.day,
        start_time.hour, start_time.minute, start_time.second
    )
    position = np.array(position)
    eci2lla = Proj(proj='latlong', datum='WGS84')
    lla = eci2lla(position[0]*1000, position[1]*1000, position[2]*1000, radians=False)


    # Imprime la posición y velocidad en el sistema de coordenadas ECI
    print("Posición ECI (X, Y, Z):", position,"[km]")
    print("Velocidad ECI (X_dot, Y_dot, Z_dot):", velocity,"[km/s]")
    print("Latitud:", lla[0],"[°]")
    print("Longitud:", lla[1],"[°]")
    #print("Altitud:", lla[2],"[m]")
