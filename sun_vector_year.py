# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 13:10:20 2023

@author: nachi
"""

#%%
from datetime import datetime, timedelta
from sgp4.earth_gravity import wgs84
from sgp4.io import twoline2rv
import matplotlib.pyplot as plt
import numpy as np

#%%
def datetime_to_jd2000(fecha):
    t1 = (fecha - datetime(2000, 1, 1, 12, 0, 0)).total_seconds()
    t2 = 86400  # Número de segundos en un día
    jd2000 = t1 / t2
    return jd2000
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
    propagation_time =12*30*24*60*60

    # Inicializa las listas para almacenar los valores a lo largo del tiempo
    times = []
    x_sun= []
    y_sun = []
    z_sun = []
    # Inicializa el tiempo actual
    current_time = start_time

    # Realiza la propagación y almacena los valores en las listas
    while current_time < start_time + timedelta(seconds=propagation_time):

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
        
        
        
        times.append(current_time)
        x_sun.append(X_sun)
        y_sun.append(Y_sun)
        z_sun.append(Z_sun)
        
        # Incrementa el tiempo actual en un paso de tiempo (por ejemplo, 1 segundo)
        current_time += timedelta(seconds=86400)

    # Convierte las listas en matrices NumPy para facilitar la manipulación

    x_sun = np.array(x_sun)
    y_sun = np.array(y_sun)
    z_sun = np.array(z_sun)


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