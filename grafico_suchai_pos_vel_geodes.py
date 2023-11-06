# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 16:04:35 2023

@author: nachi
"""
#%%
import matplotlib.pyplot as plt
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

#%%
def datetime_to_jd2000(fecha):
    t1 = (fecha - datetime(2000, 1, 1, 12, 0, 0)).total_seconds()
    t2 = 86400  # Número de segundos en un día
    jd2000 = t1 / t2
    return jd2000
#%%
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
    start_time = datetime(2023, 7, 5, 12, 0, 0)  # Ejemplo: 5 de julio de 2023, 12:00:00

    # Define el tiempo de propagación en segundos 
    propagation_time = 24*60*60

    # Inicializa las listas para almacenar los valores a lo largo del tiempo
    times = []
    positions = []
    velocities = []
    latitudes = []
    longitudes = []
    altitudes = []
    Bx = []
    By = []
    Bz = []
    x_sun= []
    y_sun = []
    z_sun = []
    B_bodys = []
    vsun_bodys = []
    # Inicializa el tiempo actual
    current_time = start_time

    # Realiza la propagación y almacena los valores en las listas
    while current_time < start_time + timedelta(seconds=propagation_time):
        position, velocity = satellite.propagate(
            current_time.year, current_time.month, current_time.day,
            current_time.hour, current_time.minute, current_time.second
        )
    
        #Para transformar el vector a LLA 
        current_time_gps = datetime(2023, 7, 5, 12, 0, 0,tzinfo=utc)  # Ejemplo: 5 de julio de 2023, 12:00:00
        lla = eci2lla(position,current_time_gps)
        
        #Obtener fuerzas magneticas de la Tierra
        B = pyIGRF.igrf_value(lla[0],lla[1],lla[2], current_time.year)
        #Fuerza magnetica en bodyframe
        q_b = np.array([0.9239, 0.2887, 0.2887, 0.2887]) #cuaternion activo q = [cos(45/2), sin(45/2)*1/raiz(3),sin(45/2)*1/raiz(3),sin(45/2)*1/raiz(3)] normalizado
        B_eci = [0,B[3],B[4],B[5]]
        inv_q_b = inv_q(q_b)
        B_body = q_b*B_eci*inv_q_b
        B_body = np.array([B_body[1],B_body[2],B_body[3]]) 
        
        #obtener vector sol ECI
        jd2000 = datetime_to_jd2000(current_time)
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
        
        
        times.append(current_time)
        positions.append(position)
        velocities.append(velocity)
        latitudes.append(lla[0])
        longitudes.append(lla[1])
        altitudes.append(lla[2])
        Bx.append(B[3])
        By.append(B[4])
        Bz.append(B[5])
        x_sun.append(X_sun)
        y_sun.append(Y_sun)
        z_sun.append(Z_sun)
        B_bodys.append(B_body)
        vsun_bodys.append(vsun_body)
        # Incrementa el tiempo actual en un paso de tiempo (por ejemplo, 1 segundo)
        current_time += timedelta(seconds=1)

    # Convierte las listas en matrices NumPy para facilitar la manipulación
    times = np.array(times)
    positions = np.array(positions)
    velocities = np.array(velocities)
    latitudes = np.array(latitudes)
    longitudes = np.array(longitudes)
    altitudes = np.array(altitudes)
    Bx = np.array(Bx)
    By = np.array(By)
    Bz = np.array(Bz)
    x_sun = np.array(x_sun)
    y_sun = np.array(y_sun)
    z_sun = np.array(z_sun)
    B_bodys = np.array(B_bodys)
    vsun_bodys = np.array(vsun_bodys)

    #%%
    # Grafica la posición a lo largo del tiempo
    plt.figure(figsize=(12, 6))
    plt.plot(times, positions[:, 0], label='Posición en X')
    plt.plot(times, positions[:, 1], label='Posición en Y')
    plt.plot(times, positions[:, 2], label='Posición en Z')
    plt.xlabel('Tiempo')
    plt.ylabel('Posición (ECI) [m]')
    plt.legend()
    plt.title('Posición del satélite a lo largo del tiempo')
    plt.grid()
    plt.show()

    # Grafica la velocidad a lo largo del tiempo
    plt.figure(figsize=(12, 6))
    plt.plot(times, velocities[:, 0], label='Velocidad en X')
    plt.plot(times, velocities[:, 1], label='Velocidad en Y')
    plt.plot(times, velocities[:, 2], label='Velocidad en Z')
    plt.xlabel('Tiempo')
    plt.ylabel('Velocidad (ECI) [m/s]')
    plt.legend()
    plt.title('Velocidad del satélite a lo largo del tiempo')
    plt.grid()
    plt.show()
    
    #%%
    # Grafica la longitud y latitud a lo largo del tiempo
    plt.figure(figsize=(12, 6))
    plt.plot(times,latitudes, label='latitud')
    plt.plot(times, longitudes, label='longitud')
    plt.xlabel('Tiempo')
    plt.ylabel('geodesicas [°]')
    plt.legend()
    plt.title('geodesicas del satélite a lo largo del tiempo')
    plt.grid()
    plt.show()

    # Grafica la altitud a lo largo del tiempo
    plt.figure(figsize=(12, 6))
    plt.plot(times, altitudes, label='altitud')
    plt.xlabel('Tiempo')
    plt.ylabel('geodesicas')
    plt.legend()
    plt.title('geodesicas del satélite a lo largo del tiempo')
    plt.grid()
    plt.show()
    #%%
    # Grafica fuerzas magnetica ECI a lo largo del tiempo
    plt.figure(figsize=(12, 6))
    plt.plot(times, Bx, label='Fuerza magnetica en X')
    plt.plot(times, By, label='Fuerza magnetica en Y')
    plt.plot(times, Bz, label='Fuerza magnetica en Z')
    plt.xlabel('Tiempo')
    plt.ylabel('Fuerza magnetica [nT]')
    plt.legend()
    plt.title('Fuerzas magneticas a lo largo del tiempo')
    plt.grid()
    plt.show()
    #%%
    # Grafica fuerzas magnetica body a lo largo del tiempo
    plt.figure(figsize=(12, 6))
    plt.plot(times, B_bodys[:,0], label='Fuerza magnetica en X')
    plt.plot(times, B_bodys[:,1], label='Fuerza magnetica en Y')
    plt.plot(times, B_bodys[:,2], label='Fuerza magnetica en Z')
    plt.xlabel('Tiempo')
    plt.ylabel('Fuerza magnetica [nT]')
    plt.legend()
    plt.title('Fuerzas magneticas en body a lo largo del tiempo')
    plt.grid()
    plt.show()
    #%%
    # Grafica vector sol ECI a lo largo del tiempo
    plt.figure(figsize=(12, 6))
    plt.plot(times, x_sun, label='Componente X vector sol ECI')
    plt.plot(times, y_sun, label='Componente Y vector sol ECI')
    plt.plot(times, z_sun, label='Componente Z vector sol ECI:')
    plt.xlabel('Tiempo')
    plt.ylabel('Vector sol')
    plt.legend()
    plt.title('vector sol a lo largo del tiempo')
    plt.grid()
    plt.show()
    #%%
    # Grafica vector sol body a lo largo del tiempo
    plt.figure(figsize=(12, 6))
    plt.plot(times, vsun_bodys[:,0], label='Componente X vector sol body')
    plt.plot(times, vsun_bodys[:,1], label='Componente Y vector sol body')
    plt.plot(times, vsun_bodys[:,2], label='Componente Z vector sol body:')
    plt.xlabel('Tiempo')
    plt.ylabel('Vector sol')
    plt.legend()
    plt.title('vector sol a lo largo del tiempo')
    plt.grid()
    plt.show()


