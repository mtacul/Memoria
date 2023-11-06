# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 04:05:44 2023

@author: nachi
"""

#%%
from skyfield.positionlib import Geocentric
from skyfield.api import utc
from datetime import datetime, timedelta
from sgp4.earth_gravity import wgs84
from sgp4.io import twoline2rv
import pyIGRF
import numpy as np

#%% funcion de eci a lla (GPS)
def eci2lla(posicion, fecha):
    from skyfield.api import Distance, load, utc, wgs84
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

#%% Funcion para pasar a JD2000
def datetime_to_jd2000(fecha):
    t1 = (fecha - datetime(2000, 1, 1, 12, 0, 0)).total_seconds()
    t2 = 86400  # Número de segundos en un día
    jd2000 = t1 / t2
    return jd2000

#%% inversa de un cuaternion
def inv_q(q):
    inv_q = np.array([q[0],-q[1],-q[2],-q[3]])
    return inv_q
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
    
    # para transformar el vector a LLA 
    start_time_gps = datetime(2023, 7, 5, 12, 0, 0,tzinfo=utc)  # Ejemplo: 5 de julio de 2023, 12:00:00
    lla = eci2lla(position,start_time_gps)

    
    #Obtener campo magnetico
    B = pyIGRF.igrf_value(lla[0], lla[1], lla[2], start_time.year)
    
    #Fuerza magnetica en bodyframe
    q_b = np.array([0.9239, 0.2887, 0.2887, 0.2887]) #cuaternion activo q = [cos(45/2), sin(45/2)*1/raiz(3),sin(45/2)*1/raiz(3),sin(45/2)*1/raiz(3)] normalizado
    B_eci = [0,B[3],B[4],B[5]]
    inv_q_b = inv_q(q_b)
    B_body = q_b*B_eci*inv_q_b
    B_body = np.array([B_body[1],B_body[2],B_body[3]])  
    
    #obtener vector sol
    jd2000 = datetime_to_jd2000(start_time)
    M_sun = 357.528 + 0.9856003*jd2000
    M_sun_rad = M_sun * np.pi/180
    lambda_sun = 280.461 + 0.9856474*jd2000 + 1.915*np.sin(M_sun_rad)+0.020*np.sin(2*M_sun_rad)
    lambda_sun_rad = lambda_sun * np.pi/180
    epsilon_sun = 23.4393 - 0.0000004*jd2000
    epsilon_sun_rad = epsilon_sun * np.pi/180

    X_sun = np.cos(lambda_sun_rad)
    Y_sun = np.cos(epsilon_sun_rad)*np.sin(lambda_sun_rad)
    Z_sun = np.sin(epsilon_sun_rad)*np.sin(lambda_sun_rad)
    
    #vector sol en body
    q_s = np.array([0.9239, 0.2887, 0.2887, 0.2887]) #cuaternion activo q = [cos(45/2), sin(45/2)*1/raiz(3),sin(45/2)*1/raiz(3),sin(45/2)*1/raiz(3)] normalizado
    vsun_eci = [0,X_sun,Y_sun,Z_sun]
    inv_q_s = inv_q(q_s)
    vsun_body = q_s*vsun_eci*inv_q_s
    vsun_body = np.array([vsun_body[1],vsun_body[2],vsun_body[3]])  

    # Imprime la posición y velocidad en el sistema de coordenadas ECI
    print("Posición ECI (X, Y, Z):", position,"[km]")
    print("Velocidad ECI (X_dot, Y_dot, Z_dot):", velocity,"[km/s]")
    print("Latitud:", lla[0],"[°]")
    print("Longitud:", lla[1],"[°]")
    print("Altitud:",lla[2],"[m] above WGS84 mean sea level")
    print("Componente X fuerza magnetica:", B[3])
    print("Componente Y fuerza magnetica:", B[4])
    print("Componente Z fuerza magnetica:", B[5])
    print("Componentes X,Y y Z en body de las fuerzas magneticas:", B_body)
    print("Componente X vector sol ECI:", X_sun)
    print("Componente Y vector sol ECI:", Y_sun)
    print("Componente Z vector sol ECI:", Z_sun)
    print("Componentes X,Y y Z en body del vector sol:", vsun_body)
