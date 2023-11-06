# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 16:04:35 2023

@author: nachi
"""
from datetime import datetime, timedelta
from sgp4.earth_gravity import wgs84
from sgp4.io import twoline2rv
import matplotlib.pyplot as plt

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

    # Inicializa el tiempo actual
    current_time = start_time

    # Realiza la propagación y almacena los valores en las listas
    while current_time < start_time + timedelta(seconds=propagation_time):
        position, velocity = satellite.propagate(
            current_time.year, current_time.month, current_time.day,
            current_time.hour, current_time.minute, current_time.second
        )

        times.append(current_time)
        positions.append(position)
        velocities.append(velocity)

        # Incrementa el tiempo actual en un paso de tiempo (por ejemplo, 1 segundo)
        current_time += timedelta(seconds=1)

    # Convierte las listas en matrices NumPy para facilitar la manipulación
    import numpy as np
    times = np.array(times)
    positions = np.array(positions)
    velocities = np.array(velocities)

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
    
