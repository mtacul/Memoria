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
from scipy.spatial.transform import Rotation


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

#%% modelo vector sol
def sun_vector(jd2000):
    M_sun = 357.528 + 0.9856003*jd2000
    M_sun_rad = M_sun * np.pi/180
    lambda_sun = 280.461 + 0.9856474*jd2000 + 1.915*np.sin(M_sun_rad)+0.020*np.sin(2*M_sun_rad)
    lambda_sun_rad = lambda_sun * np.pi/180
    epsilon_sun = 23.4393 - 0.0000004*jd2000
    epsilon_sun_rad = epsilon_sun * np.pi/180
    X_sun = np.cos(lambda_sun_rad)
    Y_sun = np.cos(epsilon_sun_rad)*np.sin(lambda_sun_rad)
    Z_sun = np.sin(epsilon_sun_rad)*np.sin(lambda_sun_rad)
    return X_sun, Y_sun, Z_sun

#%% TRIAD solution
def TRIAD(V1,V2,W1,W2):
    r1 = V1
    r2 = np.cross(V1,V2) / np.linalg.norm(np.cross(V1,V2))
    r3 = np.cross(r1,r2)
    M_obs = np.array([r1,r2,r3])
    s1 = W1
    s2 = np.cross(W1,W2) / np.linalg.norm(np.cross(W1,W2))
    s3 = np.cross(s1,s2)
    M_ref = np.array([s1,s2,s3])
    
    A = np.dot(M_obs,np.transpose(M_ref))
    return A

#%% fecha a JD2000
def datetime_to_jd2000(fecha):
    t1 = (fecha - datetime(2000, 1, 1, 12, 0, 0)).total_seconds()
    t2 = 86400  # Número de segundos en un día
    jd2000 = t1 / t2
    return jd2000

#%%inversa de un cuaternion
def inv_q(q):
    inv_q = np.array([q[0],-q[1],-q[2],-q[3]])
    return inv_q

#%% Normalizar un cuaternión

def normalize_quaternion(q):
    # Convierte el cuaternión a un arreglo NumPy para realizar cálculos más eficientes
    q = np.array(q)
    
    # Calcula la magnitud (norma) del cuaternión
    magnitude = np.linalg.norm(q)
    
    # Evita la división por cero si el cuaternión es nulo
    if magnitude == 0:
        return np.array([0.0, 0.0, 0.0, 0.0])
    
    # Divide cada componente del cuaternión por su magnitud
    normalized_q = q / magnitude
    
    # Convierte el resultado de vuelta a una tupla
    return normalized_q

#%%Funciones correspondientes a la cinematica del cuaternion y la dinamica rotacional

def f1(t, q0, q1, q2, q3, w0, w1, w2): #q1_dot
    return 0.5*(q1*w2 - q2*w1 + q3*w0)

def f2(t, q0, q1, q2, q3, w0, w1, w2): #q2_dot
    return 0.5*(-q0*w2 + q2*w0 + q3*w1)

def f3(t, q0, q1, q2, q3, w0, w1, w2): #q3_dot
    return 0.5*(q0*w1 - q1*w0 + q3*w2)

def f4(t, q0, q1, q2, q3, w0, w1, w2): #q4_dot
    return 0.5*(-q0*w0 - q1*w1 - q2*w2)

def f5(t, q0, q1, q2, q3, w0, w1, w2):#w1_dot
    return (w1*w2*(I_y-I_z))/I_x + tau/I_x

def f6(t, q0, q1, q2, q3, w0, w1, w2): #w2_dot
    return (w0*w2*(I_x-I_z))/I_y + tau/I_y

def f7(t, q0, q1, q2, q3, w0, w1, w2): #w3_dot
    return (w0*w1*(I_x-I_y))/I_z + tau/I_z

#%% rk4 para cinematica y dinamica de actitud

def rk4_step(t, q0, q1, q2, q3, w0, w1, w2, h):
    #k1 = h * f1(x, y1, y2)
    k1_1 = h * f1(t, q0, q1, q2, q3, w0, w1, w2)
    k1_2 = h * f2(t, q0, q1, q2, q3, w0, w1, w2)
    k1_3 = h * f3(t, q0, q1, q2, q3, w0, w1, w2)
    k1_4 = h * f4(t, q0, q1, q2, q3, w0, w1, w2)
    k1_5 = h * f5(t, q0, q1, q2, q3, w0, w1, w2)
    k1_6 = h * f6(t, q0, q1, q2, q3, w0, w1, w2)
    k1_7 = h * f7(t, q0, q1, q2, q3, w0, w1, w2)
    
    k2_1 = h * f1(t + 0.5 * h, q0 + 0.5 * k1_1, q1 + 0.5 * k1_2, q2 + 0.5 * k1_3,
                  q3 + 0.5 * k1_4, w0 + 0.5 * k1_5, w1 + 0.5 * k1_6, w2 + 0.5 * k1_7)
    k2_2 = h * f2(t + 0.5 * h, q0 + 0.5 * k1_1, q1 + 0.5 * k1_2, q2 + 0.5 * k1_3,
                  q3 + 0.5 * k1_4, w0 + 0.5 * k1_5, w1 + 0.5 * k1_6, w2 + 0.5 * k1_7)
    k2_3 = h * f3(t + 0.5 * h, q0 + 0.5 * k1_1, q1 + 0.5 * k1_2, q2 + 0.5 * k1_3,
                  q3 + 0.5 * k1_4, w0 + 0.5 * k1_5, w1 + 0.5 * k1_6, w2 + 0.5 * k1_7)
    k2_4 = h * f4(t + 0.5 * h, q0 + 0.5 * k1_1, q1 + 0.5 * k1_2, q2 + 0.5 * k1_3,
                  q3 + 0.5 * k1_4, w0 + 0.5 * k1_5, w1 + 0.5 * k1_6, w2 + 0.5 * k1_7)
    k2_5 = h * f5(t + 0.5 * h, q0 + 0.5 * k1_1, q1 + 0.5 * k1_2, q2 + 0.5 * k1_3,
                  q3 + 0.5 * k1_4, w0 + 0.5 * k1_5, w1 + 0.5 * k1_6, w2 + 0.5 * k1_7)
    k2_6 = h * f6(t + 0.5 * h, q0 + 0.5 * k1_1, q1 + 0.5 * k1_2, q2 + 0.5 * k1_3,
                  q3 + 0.5 * k1_4, w0 + 0.5 * k1_5, w1 + 0.5 * k1_6, w2 + 0.5 * k1_7)
    k2_7 = h * f7(t + 0.5 * h, q0 + 0.5 * k1_1, q1 + 0.5 * k1_2, q2 + 0.5 * k1_3,
                  q3 + 0.5 * k1_4, w0 + 0.5 * k1_5, w1 + 0.5 * k1_6, w2 + 0.5 * k1_7)
    
    k3_1 = h * f1(t + 0.5 * h, q0 + 0.5 * k2_1, q1 + 0.5 * k2_2, q2 + 0.5 * k2_3,
                  q3 + 0.5 * k2_4, w0 + 0.5 * k2_5, w1 + 0.5 * k2_6, w2 + 0.5 * k2_7)
    k3_2 = h * f2(t + 0.5 * h, q0 + 0.5 * k2_1, q1 + 0.5 * k2_2, q2 + 0.5 * k2_3,
                  q3 + 0.5 * k2_4, w0 + 0.5 * k2_5, w1 + 0.5 * k2_6, w2 + 0.5 * k2_7)
    k3_3 = h * f3(t + 0.5 * h, q0 + 0.5 * k2_1, q1 + 0.5 * k2_2, q2 + 0.5 * k2_3,
                  q3 + 0.5 * k2_4, w0 + 0.5 * k2_5, w1 + 0.5 * k2_6, w2 + 0.5 * k2_7)
    k3_4 = h * f4(t + 0.5 * h, q0 + 0.5 * k1_1, q1 + 0.5 * k2_2, q2 + 0.5 * k2_3,
                  q3 + 0.5 * k2_4, w0 + 0.5 * k2_5, w1 + 0.5 * k2_6, w2 + 0.5 * k2_7)
    k3_5 = h * f5(t + 0.5 * h, q0 + 0.5 * k1_1, q1 + 0.5 * k2_2, q2 + 0.5 * k2_3,
                  q3 + 0.5 * k2_4, w0 + 0.5 * k2_5, w1 + 0.5 * k2_6, w2 + 0.5 * k2_7)
    k3_6 = h * f6(t + 0.5 * h, q0 + 0.5 * k1_1, q1 + 0.5 * k2_2, q2 + 0.5 * k2_3,
                  q3 + 0.5 * k2_4, w0 + 0.5 * k2_5, w1 + 0.5 * k2_6, w2 + 0.5 * k2_7)
    k3_7 = h * f7(t + 0.5 * h, q0 + 0.5 * k1_1, q1 + 0.5 * k2_2, q2 + 0.5 * k2_3,
                  q3 + 0.5 * k2_4, w0 + 0.5 * k2_5, w1 + 0.5 * k2_6, w2 + 0.5 * k2_7)
    
    k4_1 = h * f1(t + 0.5 * h, q0 + 0.5 * k3_1, q1 + 0.5 * k3_2, q2 + 0.5 * k3_3,
                  q3 + 0.5 * k3_4, w0 + 0.5 * k3_5, w1 + 0.5 * k3_6, w2 + 0.5 * k3_7)
    k4_2 = h * f2(t + 0.5 * h, q0 + 0.5 * k3_1, q1 + 0.5 * k3_2, q2 + 0.5 * k3_3,
                  q3 + 0.5 * k3_4, w0 + 0.5 * k3_5, w1 + 0.5 * k3_6, w2 + 0.5 * k3_7)
    k4_3 = h * f3(t + 0.5 * h, q0 + 0.5 * k3_1, q1 + 0.5 * k3_2, q2 + 0.5 * k3_3,
                  q3 + 0.5 * k3_4, w0 + 0.5 * k3_5, w1 + 0.5 * k3_6, w2 + 0.5 * k3_7)
    k4_4 = h * f4(t + 0.5 * h, q0 + 0.5 * k3_1, q1 + 0.5 * k3_2, q2 + 0.5 * k3_3,
                  q3 + 0.5 * k3_4, w0 + 0.5 * k3_5, w1 + 0.5 * k3_6, w2 + 0.5 * k3_7)
    k4_5 = h * f5(t + 0.5 * h, q0 + 0.5 * k3_1, q1 + 0.5 * k3_2, q2 + 0.5 * k3_3,
                  q3 + 0.5 * k3_4, w0 + 0.5 * k3_5, w1 + 0.5 * k3_6, w2 + 0.5 * k3_7)
    k4_6 = h * f6(t + 0.5 * h, q0 + 0.5 * k3_1, q1 + 0.5 * k3_2, q2 + 0.5 * k3_3,
                  q3 + 0.5 * k3_4, w0 + 0.5 * k3_5, w1 + 0.5 * k3_6, w2 + 0.5 * k3_7)
    k4_7 = h * f7(t + 0.5 * h, q0 + 0.5 * k3_1, q1 + 0.5 * k3_2, q2 + 0.5 * k3_3,
                  q3 + 0.5 * k3_4, w0 + 0.5 * k3_5, w1 + 0.5 * k3_6, w2 + 0.5 * k3_7)

    q0_new = q0 + (k1_1 + 2 * k2_1 + 2 * k3_1 + k4_1) / 6    
    q1_new = q1 + (k1_2 + 2 * k2_2 + 2 * k3_2 + k4_2) / 6
    q2_new = q2 + (k1_3 + 2 * k2_3 + 2 * k3_3 + k4_3) / 6
    q3_new = q3 + (k1_4 + 2 * k2_4 + 2 * k3_4 + k4_4) / 6
    q = [q0_new, q1_new, q2_new, q3_new]
    q_n = normalize_quaternion(q)
    
    w0_new = w0 + (k1_5 + 2 * k2_5 + 2 * k3_5 + k4_5) / 6
    w1_new = w1 + (k1_6 + 2 * k2_6 + 2 * k3_6 + k4_6) / 6
    w2_new = w2 + (k1_7 + 2 * k2_7 + 2 * k3_7 + k4_7) / 6
    w = [w0_new, w1_new, w2_new]

    return q_n, w
#%% de cuaternion a angulos de euler
def quaternion_to_euler(q):
    # Extracción de los componentes del cuaternión
    w, x, y, z = q

    # Cálculo de ángulos de Euler en radianes
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    pitch = np.arcsin(2 * (w * y - z * x))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

    # Convierte los ángulos a grados si lo deseas
    roll_deg = np.degrees(roll)
    pitch_deg = np.degrees(pitch)
    yaw_deg = np.degrees(yaw)

    return roll_deg, pitch_deg, yaw_deg

#%% Función para simular lecturas del magnetómetro con ruido
 
def simulate_magnetometer_reading(B_eci, rango, ruido):
    num_samples = len(B_eci)
    readings = []

    for i in range(num_samples):
        # Simular el ruido gaussiano
        noise = np.random.normal(0, ruido, 1)
        
        # Simular la medición del magnetómetro con ruido
        measurement = B_eci[i] + noise
        
        # Aplicar el rango 
        measurement = max(min(measurement, rango), -rango)
        
        readings.append(measurement)

    return readings

#%% Obtener la desviacion estandar del sun sensor

def sigma_sensor(acc):
    sigma = acc/(2*3)
    return sigma

#%% Funcion para generar realismo del sun sensor

def simulate_sunsensor_reading(lambda_s, epsilon_s ,sigma):
    
    num_samples = len(lambda_s)
    reading_lambda = []
    reading_epsilon = []
    
    for i in range(num_samples):
        
        lambda_s_d = lambda_s*180/np.pi
        epsilon_s_d = epsilon_s*180/np.pi
        # Simulación de la medición con error
        error = np.random.normal(0, sigma, 1)  # Genera un error aleatorio dentro de la precisión del sensor
        
        measured_lambda = lambda_s_d + error
        measured_epsilon = epsilon_s_d + error
        
        measured_lambda_rad = measured_lambda * np.pi/180
        measured_epsilon_rad = measured_epsilon * np.pi/180
        
        reading_lambda.append(measured_lambda_rad)
        reading_epsilon.append(measured_epsilon_rad)

    return reading_lambda, reading_epsilon

#%% Funcion para generar realismo del giroscopio

def simulate_gyros_reading(w,ruido,ARW,bias,t):
    num_samples = len(w)
    readings = []
    for i in range(num_samples):
        #aplicar el ruido del sensor
        noise = np.random.normal(0, ruido, 1)
        
        #aplicar el ARW del sensor, multiplicandolo con el tiempo
        ARW_t = ARW*np.sqrt(t[i])
        
        #Simular la medicion del giroscopio
        measurement = w[i] + noise + ARW_t + bias
        
        readings.append(measurement)
        
    return readings

#%%

I_x = 1
I_y = 0.5
I_z = 2
tau = 1e-5 #torques externos de LEO SE DEBEN METER COMO ECUACIONES

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
    propagation_time = 60*60

    # Inicializa las listas para almacenar los valores a lo largo del tiempo
    times = []
    positions = []
    velocities = []
    latitudes = []
    longitudes = []
    altitudes = []
    Bs = []
    vsun = []
    B_bodys = []
    vsun_bodys = []
    lambdas_bodys = []
    epsilon_bodys =[]
    
    #posicion y velocidad del satelite en la fecha inicial
    position_i, velocity_i = satellite.propagate(
        start_time.year, start_time.month, start_time.day,
        start_time.hour, start_time.minute, start_time.second)
    
    #Para transformar el vector a LLA inicial
    start_time_gps = datetime(2023, 7, 5, 12, 0, 0,tzinfo=utc)  # Ejemplo: 5 de julio de 2023, 12:00:00
    lla = eci2lla(position_i,start_time_gps)
    
    #Obtener fuerzas magneticas de la Tierra inicial
    Bi = pyIGRF.igrf_value(lla[0],lla[1],lla[2], start_time.year)
    Bi_f = np.array([Bi[3], Bi[4], Bi[5]])
    
    #Fuerza magnetica en bodyframe inicial
    qi_b = np.array([0, 1, 0, 0]) #cuaternion activo q = [cos(45/2), sin(45/2)*1/raiz(3),sin(45/2)*1/raiz(3),sin(45/2)*1/raiz(3)] normalizado
    Bi_eci = [0,Bi_f[0],Bi_f[1],Bi_f[2]]
    inv_qi_b = inv_q(qi_b)
    Bi_body = qi_b*Bi_eci*inv_qi_b
    Bi_body = np.array([Bi_body[1],Bi_body[2],Bi_body[3]]) 
    
    #obtener vector sol ECI inicial
    jd2000i = datetime_to_jd2000(start_time)
    sunvectori = sun_vector(jd2000i)
    
    #vector sol en body inicial
    qi_s = np.array([0, 0, 1, 0]) #cuaternion activo q = [cos(45/2), sin(45/2)*1/raiz(3),sin(45/2)*1/raiz(3),sin(45/2)*1/raiz(3)] normalizado
    vsuni_eci = [0,sunvectori[0],sunvectori[1],sunvectori[2]]
    inv_qi_s = inv_q(qi_s)
    vsuni_body = qi_s*vsuni_eci*inv_qi_s
    vsuni_body = np.array([vsuni_body[1],vsuni_body[2],vsuni_body[3]]) 
    
    #Creacion de la solucion TRIAD inicial en matriz de cosenos directores y su cuaternion
    DCM = TRIAD(Bi_body,vsuni_body,Bi_f,sunvectori)
    q = Rotation.from_matrix(DCM).as_quat()
    RPY_i = quaternion_to_euler(q)
    
    #%% sensores
    # Características del magnetómetro bueno
    rango = 800000  # nT
    ruido = 1.18  # nT/√Hz
    
    #Caracteristicas del magnetometro malo
    rango_bad = 75000 #nT
    ruido_bad = 5 #nT/√Hz
    
    #caracteristicas del sun sensor malo
    acc_bad = 5 #°
    sigma_bad = sigma_sensor(acc_bad)
    
    #Caracteristicas del sun sensor intermedio
    acc_med = 1 #°
    sigma_med = sigma_sensor(acc_med)
    
    #caracteristicas del sun sensor bueno
    sigma_good = 0.05
    
    # Datos del giroscopio malo
    bias_instability_bad = 0.10 / 3600 *np.pi/180 # <0.10°/h en radianes por segundo
    noise_rms_bad = 0.12*np.pi/180 # 0.12 °/s en radianes por segundo
    angle_random_walk_bad = 0.006*np.pi/180 *1/60 # 0.006 °/√(h) en radianes por segundo por raíz de hora
    
    #Datos del giroscopio medio
    bias_instability_med = 0.05 / 3600 * np.pi/180  # <0.06°/h en radianes por segundo
    noise_rms_med = 0.12 *np.pi/180 # 0.12 °/s rms en radianes por segundo
    angle_random_walk_med = 0.006 *np.pi/180 *1/60 # 0.006 °/√(h) en radianes por segundo por raíz de hora
    
    #Datos del giroscopio bueno
    bias_instability_good = 0.03 / 3600 *np.pi/180 # <0.06°/h en radianes por segundo
    noise_rms_good= 0.050 *np.pi/180  # 0.050 °/s rms en radianes por segundo
    angle_random_walk_good = 0.006 * np.pi/180 *1/60 # 0.006 °/√(h) en radianes por segundo por raíz de hora

    
    #%% Condiciones iniciales para q (triada) y w (velocidades angulares)
    h = 0.1
    t0 = 0
    w0_0 = 0.1
    w1_0 = 0.1
    w2_0 = 0
    t_end = propagation_time

    t_values = [t0]
    q0_values = [q[0]]
    q1_values = [q[1]]
    q2_values = [q[2]]
    q3_values = [q[3]]
    w0_values = [w0_0]
    w1_values = [w1_0]
    w2_values = [w2_0]
    RPY_values = [RPY_i]
    
    #%% Propagacion
    
    # Inicializa el tiempo actual un segundo despues del inicio
    current_time = start_time+ timedelta(seconds=1)
    current_time_gps = start_time_gps + timedelta(seconds=1)
    
    # Realiza la propagación y almacena los valores en las listas
    while current_time < start_time + timedelta(seconds=propagation_time):
        
        position, velocity = satellite.propagate(
            current_time.year, current_time.month, current_time.day,
            current_time.hour, current_time.minute, current_time.second
        )
    
        #Para transformar el vector a LLA 
        lla = eci2lla(position,current_time_gps)
        
        #Obtener fuerzas magneticas de la Tierra
        B = pyIGRF.igrf_value(lla[0],lla[1],lla[2], current_time.year)
        B_f = np.array([B[3], B[4], B[5]])
        #Fuerza magnetica en bodyframe
        q_b = np.array([0, 1, 0, 0]) #cuaternion activo q = [cos(45/2), sin(45/2)*1/raiz(3),sin(45/2)*1/raiz(3),sin(45/2)*1/raiz(3)] normalizado
        B_eci = [0,B_f[0],B_f[1],B_f[2]]
        inv_q_b = inv_q(q_b)
        B_body = q_b*B_eci*inv_q_b
        B_body = np.array([B_body[1],B_body[2],B_body[3]]) 
        
        #obtener vector sol ECI
        jd2000 = datetime_to_jd2000(current_time)
        print(jd2000)
        sunvector = sun_vector(jd2000)
        
        #vector sol en body
        q_s = np.array([0, 0, 1, 0]) #cuaternion activo q = [cos(45/2), sin(45/2)*1/raiz(3),sin(45/2)*1/raiz(3),sin(45/2)*1/raiz(3)] normalizado
        vsun_eci = [0,sunvector[0],sunvector[1],sunvector[2]]
        inv_q_s = inv_q(q_s)
        vsun_body = q_s*vsun_eci*inv_q_s
        vsun_body = np.array([vsun_body[1],vsun_body[2],vsun_body[3]]) 
        lambda_body = np.arccos(vsun_body[0])
        epsilon_body = np.arccos(vsun_body[1]/np.sin(lambda_body))
        
        # Solucion de rk4 para las rotaciones
        for i in range(int(1/h)):
            t, q0, q1, q2, q3, w0, w1, w2 = t_values[-1], q0_values[-1], q1_values[-1], q2_values[-1], q3_values[-1], w0_values[-1], w1_values[-1], w2_values[-1]
            qn_new, w_new = rk4_step(t, q0, q1, q2, q3, w0, w1, w2, h)
            RPY =quaternion_to_euler(qn_new)
        
            #valores de rotacion
            t_values.append(t + h)
            q0_values.append(qn_new[0])
            q1_values.append(qn_new[1])
            q2_values.append(qn_new[2])
            q3_values.append(qn_new[3])
            w0_values.append(w_new[0])
            w1_values.append(w_new[1])
            w2_values.append(w_new[2])
            RPY_values.append(RPY)
        
        #mediciones y posiciones
        times.append(current_time)
        positions.append(position)
        velocities.append(velocity)
        latitudes.append(lla[0])
        longitudes.append(lla[1])
        altitudes.append(lla[2])
        Bs.append(B_f)
        vsun.append(sunvector)
        B_bodys.append(B_body)
        vsun_bodys.append(vsun_body)
        lambdas_bodys.append(lambda_body)
        epsilon_bodys.append(epsilon_body)
        
        # Incrementa el tiempo actual en un paso de tiempo (por ejemplo, 1 segundo)
        current_time += timedelta(seconds=1)
        current_time_gps += timedelta(seconds=1)

    # Convierte las listas en matrices NumPy para facilitar la manipulación
    times = np.array(times)
    positions = np.array(positions)
    velocities = np.array(velocities)
    latitudes = np.array(latitudes)
    longitudes = np.array(longitudes)
    altitudes = np.array(altitudes)
    Bs = np.array(Bs)
    vsun = np.array(vsun)
    B_bodys = np.array(B_bodys)
    vsun_bodys = np.array(vsun_bodys)
    lambdas_bodys = np.array(lambdas_bodys)
    epsilon_bodys = np.array(epsilon_bodys)
    RPY_values = np.array(RPY_values)
    
    #%% Sensores con ruido
    
    #sun sensor malo con ruido
    sun_sensor_bad = simulate_sunsensor_reading(lambdas_bodys,epsilon_bodys, sigma_bad)
    sun_sensor_bad_a = np.array(sun_sensor_bad)
    lambdas_meas_bad = sun_sensor_bad_a[0,:,0]
    epsilon_meas_bad = sun_sensor_bad_a[1,:,0]
    x_sun_vector_bad = np.cos(lambdas_meas_bad)
    y_sun_vector_bad = np.cos(epsilon_meas_bad)*np.sin(lambdas_meas_bad)
    z_sun_vector_bad = np.sin(epsilon_meas_bad)*np.sin(lambdas_meas_bad)-0.45008
    
    
    #sun sensor medio con ruido
    sun_sensor_med = simulate_sunsensor_reading(lambdas_bodys,epsilon_bodys, sigma_med)
    sun_sensor_med_a = np.array(sun_sensor_med)
    lambdas_meas_med = sun_sensor_med_a[0,:,0]
    epsilon_meas_med = sun_sensor_med_a[1,:,0]
    x_sun_vector_med = np.cos(lambdas_meas_med)
    y_sun_vector_med = np.cos(epsilon_meas_med)*np.sin(lambdas_meas_med)
    z_sun_vector_med = np.sin(epsilon_meas_med)*np.sin(lambdas_meas_med)-0.45008
    
    #sun sensor bueno con ruido
    sun_sensor_good = simulate_sunsensor_reading(lambdas_bodys,epsilon_bodys, sigma_good)
    sun_sensor_good_a = np.array(sun_sensor_good)
    lambdas_meas_good = sun_sensor_good_a[0,:,0]
    epsilon_meas_good = sun_sensor_good_a[1,:,0]
    x_sun_vector_good = np.cos(lambdas_meas_good)
    y_sun_vector_good = np.cos(epsilon_meas_good)*np.sin(lambdas_meas_good)
    z_sun_vector_good = np.sin(epsilon_meas_good)*np.sin(lambdas_meas_good)-0.45008
    
    # magnetometro bueno con ruido
    Bs_magn_x = simulate_magnetometer_reading(B_bodys[:,0], rango, ruido)
    Bs_magn_x = np.array(Bs_magn_x)
    Bs_magn_y = simulate_magnetometer_reading(B_bodys[:,1], rango, ruido)
    Bs_magn_y = np.array(Bs_magn_y)
    Bs_magn_z = simulate_magnetometer_reading(B_bodys[:,2], rango, ruido)
    Bs_magn_z = np.array(Bs_magn_z)
    
    # magnetometro malo con ruido
    Bb_magn_x = simulate_magnetometer_reading(B_bodys[:,0], rango_bad, ruido_bad)
    Bb_magn_x = np.array(Bb_magn_x)
    Bb_magn_y = simulate_magnetometer_reading(B_bodys[:,1], rango_bad, ruido_bad)
    Bb_magn_y = np.array(Bb_magn_y)
    Bb_magn_z = simulate_magnetometer_reading(B_bodys[:,2], rango_bad, ruido_bad)
    Bb_magn_z = np.array(Bb_magn_z)
    
    #giroscopo malo con ruido
    wx_gyros_bad = simulate_gyros_reading(w0_values,noise_rms_bad,angle_random_walk_bad,bias_instability_bad,t_values)
    wy_gyros_bad = simulate_gyros_reading(w1_values,noise_rms_bad,angle_random_walk_bad,bias_instability_bad,t_values)
    wz_gyros_bad = simulate_gyros_reading(w2_values,noise_rms_bad,angle_random_walk_bad,bias_instability_bad,t_values)

    #giroscopo medio con ruido
    wx_gyros_med = simulate_gyros_reading(w0_values,noise_rms_med,angle_random_walk_med,bias_instability_med,t_values)
    wy_gyros_med = simulate_gyros_reading(w1_values,noise_rms_med,angle_random_walk_med,bias_instability_med,t_values)
    wz_gyros_med = simulate_gyros_reading(w2_values,noise_rms_med,angle_random_walk_med,bias_instability_med,t_values)
    
    #giroscopo bueno con ruido
    wx_gyros_good = simulate_gyros_reading(w0_values,noise_rms_good,angle_random_walk_good,bias_instability_good,t_values)
    wy_gyros_good = simulate_gyros_reading(w1_values,noise_rms_good,angle_random_walk_good,bias_instability_good,t_values)
    wz_gyros_good = simulate_gyros_reading(w2_values,noise_rms_good,angle_random_walk_good,bias_instability_good,t_values)


    #%% graficas obtenidas por SGP4
    
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
    
    #%% GRAFICAS DE COORDENADAS GPS
    
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
    
    #%% GRAFICAS DE IGRF
    
    # Grafica fuerzas magnetica ECI a lo largo del tiempo
    plt.figure(figsize=(12, 6))
    plt.plot(times, Bs[:,0], label='Fuerza magnetica en X')
    plt.plot(times, Bs[:,1], label='Fuerza magnetica en Y')
    plt.plot(times, Bs[:,2], label='Fuerza magnetica en Z')
    plt.xlabel('Tiempo')
    plt.ylabel('Fuerza magnetica [nT]')
    plt.legend()
    plt.title('Fuerzas magneticas a lo largo del tiempo')
    plt.grid()
    plt.show()
    #%% GRAFICAS DE FUERZAS MAGNETICAS ROTADAS
    
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
    #%% GRAFICAS VECTOR SOL
    
    # Grafica vector sol ECI a lo largo del tiempo
    plt.figure(figsize=(12, 6))
    plt.plot(times, vsun[:,0], label='Componente X vector sol ECI')
    plt.plot(times, vsun[:,1], label='Componente Y vector sol ECI')
    plt.plot(times, vsun[:,2], label='Componente Z vector sol ECI:')
    plt.xlabel('Tiempo')
    plt.ylabel('Vector sol')
    plt.legend()
    plt.title('vector sol a lo largo del tiempo')
    plt.grid()
    plt.show()
    #%% GRAFICAS VECTOR SOL ROTADAS
    
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
    
    plt.figure(figsize=(12, 6))
    plt.plot(times, lambdas_bodys, label='lambdas body en sun sensor')
    plt.xlabel('Tiempo')
    plt.ylabel('lambda [rad]')
    plt.legend()
    plt.title('lambdas en body a lo largo del tiempo')
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(times, epsilon_bodys, label='epsilon body en sun sensor')
    plt.xlabel('Tiempo')
    plt.ylabel('epsilon [rad]')
    plt.legend()
    plt.title('epsilon en body a lo largo del tiempo')
    plt.grid()
    plt.show()
    
    #%% orientaciones y velocidades angulares
    
    plt.plot(t_values, q0_values, label='q0(t)')
    plt.plot(t_values, q1_values, label='q1(t)')
    plt.plot(t_values, q2_values, label='q2(t)')
    plt.plot(t_values, q3_values, label='q3(t)')
    plt.xlabel('t [s]')
    plt.ylabel('cuaternión')
    plt.title('cuaterniones vs tiempo')
    plt.grid()
    plt.legend()
    plt.show()

    plt.plot(t_values, w0_values, label='w0(t)')
    plt.plot(t_values, w1_values, label='w1(t)')
    plt.plot(t_values, w2_values, label='w2(t)')
    plt.xlabel('t [s]')
    plt.ylabel('Vel. angular [rad/s]')
    plt.title('Velocidad angular vs tiempo')
    plt.grid()
    plt.legend()
    plt.show()

    plt.plot(t_values, RPY_values[:,0], label='roll [°]')
    plt.plot(t_values, RPY_values[:,1], label='pitch [°]')
    plt.plot(t_values, RPY_values[:,2], label='yaw [°]')
    plt.xlabel('t [s]')
    plt.ylabel('Angulos de Euler [°]')
    plt.title('Angulos de Euler vs tiempo')
    plt.grid()
    plt.legend()
    plt.show()

    plt.plot(t_values, RPY_values[:,0], label='roll [°]')
    plt.plot(t_values, RPY_values[:,1], label='pitch [°]')
    plt.xlabel('t [s]')
    plt.ylabel('Angulos de Euler [°]')
    plt.title('Roll y pitch vs tiempo')
    plt.grid()
    plt.legend()
    plt.show()
    

    plt.plot(t_values, RPY_values[:,2], label='yaw [°]')
    plt.xlabel('t [s]')
    plt.ylabel('Angulos de Euler [°]')
    plt.title('Yaw vs tiempo')
    plt.grid()
    plt.legend()
    plt.show()
    
    #%% MAGNETOMETRO
    
    # Grafica fuerzas magnetica sensor bueno a lo largo del tiempo
    plt.figure(figsize=(12, 6))
    plt.plot(times, Bs_magn_x[:,0], label='Fuerza magnetica en X')
    plt.plot(times, Bs_magn_y[:,0], label='Fuerza magnetica en Y')
    plt.plot(times, Bs_magn_z[:,0], label='Fuerza magnetica en Z')
    plt.xlabel('Tiempo')
    plt.ylabel('Fuerza magnetica [nT]')
    plt.legend()
    plt.title('Fuerzas magneticas segun sensor bueno a lo largo del tiempo')
    plt.grid()
    plt.show()
    
    # Grafica fuerzas magnetica sensor malo a lo largo del tiempo
    plt.figure(figsize=(12, 6))
    plt.plot(times, Bb_magn_x[:,0], label='Fuerza magnetica en X')
    plt.plot(times, Bb_magn_y[:,0], label='Fuerza magnetica en Y')
    plt.plot(times, Bb_magn_z[:,0], label='Fuerza magnetica en Z')
    plt.xlabel('Tiempo')
    plt.ylabel('Fuerza magnetica [nT]')
    plt.legend()
    plt.title('Fuerzas magneticas segun sensor malo a lo largo del tiempo')
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(times, Bb_magn_x[:,0], label='Fuerza magnetica en X sensor malo')
    plt.plot(times, Bs_magn_x[:,0], label='Fuerza magnetica en X sensor bueno')
    plt.plot(times, B_bodys[:,0], label='Fuerza magnetica en X ideal')
    plt.xlabel('Tiempo')
    plt.ylabel('Fuerza magnetica [nT]')
    plt.legend()
    plt.title('Fuerzas magneticas a lo largo del tiempo')
    plt.grid()
    plt.show()
    
    #%% SUN SENSOR

    # Graficas sun sensor malo
    plt.figure(figsize=(12, 6))
    plt.plot(times, lambdas_meas_bad, label='lambdas medidas en sun sensor')
    plt.plot(times, lambdas_bodys, label='lambdas body en sun sensor')
    plt.xlabel('Tiempo')
    plt.ylabel('lambda [rad]')
    plt.legend()
    plt.title('lambdas segun sun sensor malo a lo largo del tiempo')
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(times, epsilon_meas_bad, label='epsilon medidas en sun sensor')
    plt.plot(times, epsilon_bodys, label='epsilon body en sun sensor')
    plt.xlabel('Tiempo')
    plt.ylabel('epsilon [rad]')
    plt.legend()
    plt.title('epsilon segun sun sensor malo a lo largo del tiempo')
    plt.grid()
    plt.show()
    
    # Graficas sun sensor medio
    plt.figure(figsize=(12, 6))
    plt.plot(times, lambdas_meas_med, label='lambdas medidas en sun sensor')
    plt.plot(times, lambdas_bodys, label='lambdas body en sun sensor')
    plt.xlabel('Tiempo')
    plt.ylabel('lambda [rad]')
    plt.legend()
    plt.title('lambdas segun sun sensor medio a lo largo del tiempo')
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(times, epsilon_meas_med, label='epsilon medidas en sun sensor')
    plt.plot(times, epsilon_bodys, label='epsilon body en sun sensor')
    plt.xlabel('Tiempo')
    plt.ylabel('epsilon [rad]')
    plt.legend()
    plt.title('epsilon segun sun sensor medio a lo largo del tiempo')
    plt.grid()
    plt.show()
    
    # Graficas sun sensor bueno
    plt.figure(figsize=(12, 6))
    plt.plot(times, lambdas_meas_good, label='lambdas medidas en sun sensor')
    plt.plot(times, lambdas_bodys, label='lambdas body en sun sensor')
    plt.xlabel('Tiempo')
    plt.ylabel('lambda [rad]')
    plt.legend()
    plt.title('lambdas segun sun sensor bueno a lo largo del tiempo')
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(times, epsilon_meas_good, label='epsilon medidas en sun sensor')
    plt.plot(times, epsilon_bodys, label='epsilon body en sun sensor')
    plt.xlabel('Tiempo')
    plt.ylabel('epsilon [rad]')
    plt.legend()
    plt.title('epsilon segun sun sensor bueno a lo largo del tiempo')
    plt.grid()
    plt.show()
    
    #graficas vectores de sol con sun sensor malo
    plt.figure(figsize=(12, 6))
    plt.plot(times,x_sun_vector_bad,label='Componente x vector sol body ruido malo:')
    plt.plot(times, vsun_bodys[:,0], label='Componente x vector sol body:')
    plt.xlabel('Tiempo')
    plt.ylabel('Vector sol')
    plt.legend()
    plt.title('vector sol a lo largo del tiempo')
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(times,y_sun_vector_bad,label='Componente y vector sol body ruido malo:')
    plt.plot(times, vsun_bodys[:,1], label='Componente y vector sol body:')
    plt.xlabel('Tiempo')
    plt.ylabel('Vector sol')
    plt.legend()
    plt.title('vector sol a lo largo del tiempo')
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(times,z_sun_vector_bad,label='Componente Z vector sol body ruido malo:')
    plt.plot(times, vsun_bodys[:,2], label='Componente Z vector sol body:')
    plt.xlabel('Tiempo')
    plt.ylabel('Vector sol')
    plt.legend()
    plt.title('vector sol a lo largo del tiempo')
    plt.grid()
    plt.show()
    
    #graficas vectores de sol con sun sensor medio
    plt.figure(figsize=(12, 6))
    plt.plot(times,x_sun_vector_med,label='Componente x vector sol body ruido medio:')
    plt.plot(times, vsun_bodys[:,0], label='Componente x vector sol body:')
    plt.xlabel('Tiempo')
    plt.ylabel('Vector sol')
    plt.legend()
    plt.title('vector sol a lo largo del tiempo')
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(times,y_sun_vector_med,label='Componente y vector sol body ruido medio:')
    plt.plot(times, vsun_bodys[:,1], label='Componente y vector sol body:')
    plt.xlabel('Tiempo')
    plt.ylabel('Vector sol')
    plt.legend()
    plt.title('vector sol a lo largo del tiempo')
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(times,z_sun_vector_med,label='Componente Z vector sol body ruido medio:')
    plt.plot(times, vsun_bodys[:,2], label='Componente Z vector sol body:')
    plt.xlabel('Tiempo')
    plt.ylabel('Vector sol')
    plt.legend()
    plt.title('vector sol a lo largo del tiempo')
    plt.grid()
    plt.show()
    
    
    #graficas vectores de sol con sun sensor bueno
    plt.figure(figsize=(12, 6))
    plt.plot(times,x_sun_vector_good,label='Componente x vector sol body ruido bueno:')
    plt.plot(times, vsun_bodys[:,0], label='Componente x vector sol body:')
    plt.xlabel('Tiempo')
    plt.ylabel('Vector sol')
    plt.legend()
    plt.title('vector sol a lo largo del tiempo')
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(times,y_sun_vector_good,label='Componente y vector sol body ruido bueno:')
    plt.plot(times, vsun_bodys[:,1], label='Componente y vector sol body:')
    plt.xlabel('Tiempo')
    plt.ylabel('Vector sol')
    plt.legend()
    plt.title('vector sol a lo largo del tiempo')
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(times,z_sun_vector_good,label='Componente Z vector sol body ruido bueno:')
    plt.plot(times, vsun_bodys[:,2], label='Componente Z vector sol body:')
    plt.xlabel('Tiempo')
    plt.ylabel('Vector sol')
    plt.legend()
    plt.title('vector sol a lo largo del tiempo')
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(times,x_sun_vector_bad,label='Componente x vector sol body ruido malo:')
    plt.plot(times,x_sun_vector_med,label='Componente x vector sol body ruido medio:')
    plt.plot(times,x_sun_vector_good,label='Componente x vector sol body ruido bueno:')
    plt.plot(times, vsun_bodys[:,0], label='Componente x vector sol body:')
    plt.xlabel('Tiempo')
    plt.ylabel('Vector sol')
    plt.legend()
    plt.title('vector sol a lo largo del tiempo')
    plt.grid()
    plt.show()
    #%% GIROSCOPIO
    
    #sensor malo
    plt.figure(figsize=(12, 6))
    plt.plot(t_values, wx_gyros_bad, label='velocidad angular en x')
    plt.plot(t_values, wy_gyros_bad, label='velocidad angular en y')
    plt.plot(t_values, wz_gyros_bad, label='velocidad angular en z')
    plt.xlabel('Tiempo')
    plt.ylabel('velocidad angular [rad/s]')
    plt.legend()
    plt.title('datos del giroscopio malo vs tiempo')
    plt.grid()
    plt.show()
    
    #sensor medio
    plt.figure(figsize=(12, 6))
    plt.plot(t_values, wx_gyros_med, label='velocidad angular en x')
    plt.plot(t_values, wy_gyros_med, label='velocidad angular en y')
    plt.plot(t_values, wz_gyros_med, label='velocidad angular en z')
    plt.xlabel('Tiempo')
    plt.ylabel('velocidad angular [rad/s]')
    plt.legend()
    plt.title('datos del giroscopio medio vs tiempo')
    plt.grid()
    plt.show()
    
    #sensor bueno
    plt.figure(figsize=(12, 6))
    plt.plot(t_values, wx_gyros_good, label='velocidad angular en x')
    plt.plot(t_values, wy_gyros_good, label='velocidad angular en y')
    plt.plot(t_values, wz_gyros_good, label='velocidad angular en z')
    plt.xlabel('Tiempo')
    plt.ylabel('velocidad angular [rad/s]')
    plt.legend()
    plt.title('datos del giroscopio bueno vs tiempo')
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(t_values, w0_values, label='velocidad angular en x modelo')
    plt.plot(t_values, wx_gyros_bad, label='velocidad angular en x BAD')
    plt.plot(t_values, wx_gyros_med, label='velocidad angular en x MED')
    plt.plot(t_values, wx_gyros_good, label='velocidad angular en x GOOD')
    plt.xlabel('Tiempo')
    plt.ylabel('velocidad angular [rad/s]')
    plt.legend()
    plt.title('datos del giroscopio e ideal vs tiempo')
    plt.grid()
    plt.show()
