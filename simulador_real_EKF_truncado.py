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

def simulate_sunsensor_reading(vsun,sigma):
    
    num_samples_l = len(vsun)
    reading_sun = []
    sigma_rad = sigma*np.pi/180
    
    for i in range(num_samples_l):
        
        # Simulación de la medición con error
        error = np.random.normal(0, sigma_rad, 1)  # Genera un error aleatorio dentro de la precisión del sensor
        
        measured_vsun = vsun[i] + error

        
        reading_sun.append(measured_vsun)

    return reading_sun

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

#%% EKF para el simulador evitando singularidad

def f(dt, q0, q1, q2,q3, w0, w1, w2):
    q0_k = 0.5*(q1*w2 - q2*w1 + q3*w0)*dt
    q1_k = 0.5*(-q0*w2 + q2*w0 + q3*w1)*dt
    q2_k = 0.5*(q0*w1 - q1*w0 + q3*w2)*dt
    q3_k = 0.5*(-q0*w0 - q1*w1 - q2*w2)*dt
    w0_k = ((w1*w2*(I_y-I_z))/I_x + tau/I_x)*dt
    w1_k = ((w0*w2*(I_x-I_z))/I_y + tau/I_y)*dt
    w2_k = ((w0*w1*(I_x-I_y))/I_z + tau/I_z)*dt
    
    return q0_k,q1_k,q2_k,q3_k,w0_k,w1_k,w2_k #entrega el X_k

#%% EKF para el simulador si q3_k = qc
def Fq3(q0_k, q1_k, q2_k,q3_k, w0_k, w1_k, w2_k):
    
    F1 = np.array([0-(q0_k/q3_k)*(0.5*w0_k), 0.5*w2_k-(q1_k/q3_k)*(0.5*w0_k), -0.5*w1_k-(q2_k/q3_k)*(0.5*w0_k), 0.5*q3_k, -0.5*q2_k, 0.5*q1_k])

    F2 = np.array([-0.5*w2_k-(q0_k/q3_k)*(0.5*w1_k), 0-(q1_k/q3_k)*(0.5*w1_k), 0.5*w0_k-(q2_k/q3_k)*(0.5*w1_k), 0.5*q2_k, 0.5*q3_k, -0.5*q0_k])

    F3 = np.array([0.5*w1_k-(q0_k/q3_k)*(0.5*w2_k), -0.5*w0_k-(q1_k/q3_k)*(0.5*w2_k), 0-(q2_k/q3_k)*(0.5*w2_k), -0.5*q1_k, 0.5*q0_k, 0.5*q3_k])
   
    F4 = np.array([0, 0, 0, 0, w2_k*(I_x-I_z)/I_y + tau/I_y, w1_k*(I_x-I_y)/I_z + tau/I_z])

    F5 = np.array([0, 0, 0, w2_k*(I_y-I_z)/I_x + tau/I_x, 0, w0_k*(I_y-I_z)/I_x + tau/I_x])

    F6 = np.array([0, 0, 0, w1_k*(I_x-I_y)/I_z + tau/I_z, w0_k*(I_x-I_y)/I_z + tau/I_z,0])

    F_k = np.array([F1,F2,F3,F4,F5,F6]) #JACOBIANO DEL VECTOR ESTADO
    print("este es de q3")
    #phi_k = np.exp(F_k) #PAPER ME DICE QUE LO DEJE ASI
    
    return F_k

def h_Xq3(q0_k, q1_k, q2_k,q3_k, w0_k, w1_k, w2_k,Bref):
    
    Bref_norm = Bref/np.linalg.norm(Bref)
    Bx_mod = (q3_k**2+q0_k**2-q1_k**2-q2_k**2)*Bref_norm[0]+ 2*(q0_k*q1_k+q2_k*q3_k)*Bref_norm[1] + 2*(q0_k*q2_k+q1_k*q3_k)*Bref_norm[2]
    By_mod = 2*(q0_k*q1_k-q2_k*q3_k)*Bref_norm[0] + (q3_k**2-q0_k**2+q1_k**2-q2_k**2)*Bref_norm[1] + 2*(q1_k*q2_k+q0_k*q3_k)*Bref_norm[2]
    Bz_mod = 2*(q0_k*q2_k+q1_k*q3_k)*Bref_norm[0] + 2*(q1_k*q2_k-q0_k*q3_k)*Bref_norm[1] + (q3_k**2-q0_k**2-q1_k**2+q2_k**2)*Bref_norm[2]
    
    B_mod = np.array([Bx_mod,By_mod,Bz_mod])
    return B_mod #entrega el z_k_priori

def Hq3(q0, q1, q2,q3, w0, w1, w2, Bref):  

    Bref_norm = Bref/np.linalg.norm(Bref)
    H1 = np.array([(2*q0)*Bref_norm[0]+(2*q1)*Bref_norm[1]+(2*q2)*Bref_norm[2]-(q0/q3)*((2*q3)*Bref_norm[0]+(2*q2)*Bref_norm[1]-(2*q1)*Bref_norm[2]), -(2*q1)*Bref_norm[0]+(2*q0)*Bref_norm[1]-(2*q3)*Bref_norm[2]-(q1/q3)*((2*q3)*Bref_norm[0]+(2*q2)*Bref_norm[1]-(2*q1)*Bref_norm[2]), -(2*q2)*Bref_norm[0]+(2*q3)*Bref_norm[1]+(2*q0)*Bref_norm[2]-(q2/q3)*((2*q3)*Bref_norm[0]+(2*q2)*Bref_norm[1]-(2*q1)*Bref_norm[2]), 0, 0, 0])
    H2 = np.array([(2*q1)*Bref_norm[0]-(2*q0)*Bref_norm[1]+(2*q3)*Bref_norm[2]-(q0/q3)*(-(2*q2)*Bref_norm[0]+(2*q3)*Bref_norm[1]+(2*q0)*Bref_norm[2]), (2*q0)*Bref_norm[0]+(2*q1)*Bref_norm[1]+(2*q2)*Bref_norm[2]-(q1/q3)*(-(2*q2)*Bref_norm[0]+(2*q3)*Bref_norm[1]+(2*q0)*Bref_norm[2]), -(2*q3)*Bref_norm[0]-(2*q2)*Bref_norm[1]+(2*q1)*Bref_norm[2]-(q2/q3)*(-(2*q2)*Bref_norm[0]+(2*q3)*Bref_norm[1]+(2*q0)*Bref_norm[2]), 0, 0, 0])
    H3 = np.array([(2*q2)*Bref_norm[0]-(2*q3)*Bref_norm[1]-(2*q0)*Bref_norm[2]-(q0/q3)*((2*q1)*Bref_norm[0]-(2*q0)*Bref_norm[1]+(2*q3)*Bref_norm[2]), (2*q3)*Bref_norm[0]+(2*q2)*Bref_norm[1]-(2*q1)*Bref_norm[2]-(q1/q3)*((2*q1)*Bref_norm[0]-(2*q0)*Bref_norm[1]+(2*q3)*Bref_norm[2]), (2*q0)*Bref_norm[0]+(2*q1)*Bref_norm[1]+(2*q2)*Bref_norm[2]-(q2/q3)*((2*q1)*Bref_norm[0]-(2*q0)*Bref_norm[1]+(2*q3)*Bref_norm[2]), 0, 0, 0])
    
    H_mat = np.array([H1,H2,H3])

    return H_mat #JACOBIANO DEL MODELO DEL SENSOR (POR AHORA SOLO MAGNETOMETRO)

#%% EKF para el simulador si q2_k = qc
def Fq2(q0_k, q1_k, q2_k, q3_k, w0_k, w1_k, w2_k):
    
    F1 = np.array([0-(q0_k/q2_k)*(-0.5*w1_k), 0.5*w2_k-(q1_k/q2_k)*(-0.5*w1_k), (0.5*w0_k)-(q3_k/q2_k)*(-0.5*w1_k), 0.5*q3_k, -0.5*q2_k, 0.5*q1_k])

    F2 = np.array([-0.5*w2_k-(q0_k/q2_k)*(0.5*w0_k), 0-(q1_k/q2_k)*((0.5*w0_k)), (0.5*w1_k)-(q3_k/q2_k)*(0.5*w0_k), 0.5*q2_k, 0.5*q3_k, -0.5*q0_k])

    F3 = np.array([-0.5*w0_k-(q0_k/q2_k)*(-0.5*w2_k), -0.5*w1_k-(q1_k/q2_k)*(-0.5*w2_k), 0-(q3_k/q2_k)*(-0.5*w1_k), -0.5*q0_k, -0.5*q1_k, -0.5*q2_k])
  
    F4 = np.array([0, 0, 0, 0, w2_k*(I_x-I_z)/I_y + tau/I_y, w1_k*(I_x-I_y)/I_z + tau/I_z])

    F5 = np.array([0, 0, 0, w2_k*(I_y-I_z)/I_x + tau/I_x, 0, w0_k*(I_y-I_z)/I_x + tau/I_x])

    F6 = np.array([0, 0, 0, w1_k*(I_x-I_y)/I_z + tau/I_z, w0_k*(I_x-I_y)/I_z + tau/I_z,0])

    F_k = np.array([F1,F2,F3,F4,F5,F6]) #JACOBIANO DEL VECTOR ESTADO
    print("este es de q2")
    #phi_k = np.exp(F_k) #PAPER ME DICE QUE LO DEJE ASI
    
    return F_k

def h_Xq2(q0_k, q1_k, q2_k, q3_k, w0_k, w1_k, w2_k,Bref):
    
    Bref_norm = Bref/np.linalg.norm(Bref)
    Bx_mod = (q3_k**2+q0_k**2-q1_k**2-q2_k**2)*Bref_norm[0]+ 2*(q0_k*q1_k+q2_k*q3_k)*Bref_norm[1] + 2*(q0_k*q2_k+q1_k*q3_k)*Bref_norm[2]
    By_mod = 2*(q0_k*q1_k-q2_k*q3_k)*Bref_norm[0] + (q3_k**2-q0_k**2+q1_k**2-q2_k**2)*Bref_norm[1] + 2*(q1_k*q2_k+q0_k*q3_k)*Bref_norm[2]
    Bz_mod = 2*(q0_k*q2_k+q1_k*q3_k)*Bref_norm[0] + 2*(q1_k*q2_k-q0_k*q3_k)*Bref_norm[1] + (q3_k**2-q0_k**2-q1_k**2+q2_k**2)*Bref_norm[2]
    
    B_mod = np.array([Bx_mod,By_mod,Bz_mod])
    return B_mod #entrega el z_k_priori

def Hq2(q0, q1,q2, q3, w0, w1, w2, Bref):  

    Bref_norm = Bref/np.linalg.norm(Bref)
    H1 = np.array([(2*q0)*Bref_norm[0]+(2*q1)*Bref_norm[1]+(2*q2)*Bref_norm[2]-(q0/q2)*-(2*q2)*Bref_norm[0]+(2*q3)*Bref_norm[1]+(2*q0)*Bref_norm[2],  -(2*q1)*Bref_norm[0]+(2*q0)*Bref_norm[1]-(2*q3)*Bref_norm[2]-(q1/q2)*-(2*q2)*Bref_norm[0]+(2*q3)*Bref_norm[1]+(2*q0)*Bref_norm[2], (2*q3)*Bref_norm[0]+(2*q2)*Bref_norm[1]-(2*q1)*Bref_norm[2]-(q3/q2)*-(2*q2)*Bref_norm[0]+(2*q3)*Bref_norm[1]+(2*q0)*Bref_norm[2], 0, 0, 0])
    H2 = np.array([(2*q1)*Bref_norm[0]-(2*q0)*Bref_norm[1]+(2*q3)*Bref_norm[2]-(q0/q2)*-(2*q3)*Bref_norm[0]-(2*q2)*Bref_norm[1]+(2*q1)*Bref_norm[2],  (2*q0)*Bref_norm[0]+(2*q1)*Bref_norm[1]+(2*q2)*Bref_norm[2]-(q1/q2)*-(2*q3)*Bref_norm[0]-(2*q2)*Bref_norm[1]+(2*q1)*Bref_norm[2], -(2*q2)*Bref_norm[0]+(2*q3)*Bref_norm[1]+(2*q0)*Bref_norm[2]-(q3/q2)*-(2*q3)*Bref_norm[0]-(2*q2)*Bref_norm[1]+(2*q1)*Bref_norm[2], 0, 0, 0])
    H3 = np.array([(2*q2)*Bref_norm[0]-(2*q3)*Bref_norm[1]-(2*q0)*Bref_norm[2]-(q0/q2)*(2*q0)*Bref_norm[0]+(2*q1)*Bref_norm[1]+(2*q2)*Bref_norm[2], (2*q3)*Bref_norm[0]+(2*q2)*Bref_norm[1]-(2*q1)*Bref_norm[2]-(q1/q2)*(2*q0)*Bref_norm[0]+(2*q1)*Bref_norm[1]+(2*q2)*Bref_norm[2],  (2*q1)*Bref_norm[0]-(2*q0)*Bref_norm[1]+(2*q3)*Bref_norm[2]-(q3/q2)*(2*q0)*Bref_norm[0]+(2*q1)*Bref_norm[1]+(2*q2)*Bref_norm[2], 0, 0, 0])
    
    
    H_mat = np.array([H1,H2,H3])

    return H_mat #JACOBIANO DEL MODELO DEL SENSOR (POR AHORA SOLO MAGNETOMETRO)

#%% EKF para el simulador si q1_k = qc
def Fq1(q0_k, q1_k, q2_k, q3_k, w0_k, w1_k, w2_k):
    
    F1 = np.array([0-(q0_k/q1_k)*0.5*w2_k, -0.5*w1_k-(q2_k/q1_k)*0.5*w2_k, 0.5*w0_k-(q3_k/q1_k)*0.5*w2_k, 0.5*q3_k, -0.5*q2_k, 0.5*q1_k]) 
  
    F2 = np.array([0.5*w1_k-(q0_k/q1_k)*(-0.5*w0_k), 0-(q2_k/q1_k)*(-0.5*w0_k), 0.5*w2_k-(q3_k/q1_k)*(-0.5*w0_k), -0.5*q1_k, 0.5*q0_k, 0.5*q3_k])

    F3 = np.array([-0.5*w0_k-(q0_k/q1_k)*(-0.5*w1_k), -0.5*w2_k-(q2_k/q1_k)*(-0.5*w1_k), 0-(q3_k/q1_k)*(-0.5*w1_k), -0.5*q0_k, -0.5*q1_k, -0.5*q2_k]) 

    F4 = np.array([0, 0, 0, 0, w2_k*(I_x-I_z)/I_y + tau/I_y, w1_k*(I_x-I_y)/I_z + tau/I_z])   

    F5 = np.array([0, 0, 0, w2_k*(I_y-I_z)/I_x + tau/I_x, 0, w0_k*(I_y-I_z)/I_x + tau/I_x])    

    F6 = np.array([0, 0, 0, w1_k*(I_x-I_y)/I_z + tau/I_z, w0_k*(I_x-I_y)/I_z + tau/I_z,0])   

    F_k = np.array([F1,F2,F3,F4,F5,F6]) #JACOBIANO DEL VECTOR ESTADO
    #phi_k = np.exp(F_k) #PAPER ME DICE QUE LO DEJE ASI
    print("este es de q1")
    return F_k

def h_Xq1(q0_k,q1_k, q2_k, q3_k, w0_k, w1_k, w2_k,Bref):
    
    Bref_norm = Bref/np.linalg.norm(Bref)
    Bx_mod = (q3_k**2+q0_k**2-q1_k**2-q2_k**2)*Bref_norm[0]+ 2*(q0_k*q1_k+q2_k*q3_k)*Bref_norm[1] + 2*(q0_k*q2_k+q1_k*q3_k)*Bref_norm[2]
    By_mod = 2*(q0_k*q1_k-q2_k*q3_k)*Bref_norm[0] + (q3_k**2-q0_k**2+q1_k**2-q2_k**2)*Bref_norm[1] + 2*(q1_k*q2_k+q0_k*q3_k)*Bref_norm[2]
    Bz_mod = 2*(q0_k*q2_k+q1_k*q3_k)*Bref_norm[0] + 2*(q1_k*q2_k-q0_k*q3_k)*Bref_norm[1] + (q3_k**2-q0_k**2-q1_k**2+q2_k**2)*Bref_norm[2]
    
    B_mod = np.array([Bx_mod,By_mod,Bz_mod])
    return B_mod #entrega el z_k_priori

def Hq1(q0,q1, q2, q3, w0, w1, w2, Bref):  

    Bref_norm = Bref/np.linalg.norm(Bref)
    H1 = np.array([(2*q0)*Bref_norm[0]+(2*q1)*Bref_norm[1]+(2*q2)*Bref_norm[2]-(q0/q1)*(-(2*q1)*Bref_norm[0]+(2*q0)*Bref_norm[1]-(2*q3)*Bref_norm[2]), -(2*q2)*Bref_norm[0]+(2*q3)*Bref_norm[1]+(2*q0)*Bref_norm[2]-(q2/q1)*(-(2*q1)*Bref_norm[0]+(2*q0)*Bref_norm[1]-(2*q3)*Bref_norm[2]), (2*q3)*Bref_norm[0]+(2*q2)*Bref_norm[1]-(2*q1)*Bref_norm[2]-(q3/q1)*-(2*q1)*Bref_norm[0]+(2*q0)*Bref_norm[1]-(2*q3)*Bref_norm[2], 0, 0, 0])
    H2 = np.array([(2*q1)*Bref_norm[0]-(2*q0)*Bref_norm[1]+(2*q3)*Bref_norm[2]-(q0/q1)*((2*q0)*Bref_norm[0]+(2*q1)*Bref_norm[1]+(2*q2)*Bref_norm[2]), -(2*q3)*Bref_norm[0]-(2*q2)*Bref_norm[1]+(2*q1)*Bref_norm[2]-(q2/q1)*((2*q0)*Bref_norm[0]+(2*q1)*Bref_norm[1]+(2*q2)*Bref_norm[2]), -(2*q2)*Bref_norm[0]+(2*q3)*Bref_norm[1]+(2*q0)*Bref_norm[2]-(q3/q1)*((2*q0)*Bref_norm[0]+(2*q1)*Bref_norm[1]+(2*q2)*Bref_norm[2]), 0, 0, 0])
    H3 = np.array([(2*q2)*Bref_norm[0]-(2*q3)*Bref_norm[1]-(2*q0)*Bref_norm[2]-(q0/q1)*((2*q3)*Bref_norm[0]+(2*q2)*Bref_norm[1]-(2*q1)*Bref_norm[2]), (2*q0)*Bref_norm[0]+(2*q1)*Bref_norm[1]+(2*q2)*Bref_norm[2]-(q2/q1)*(2*q3)*Bref_norm[0]+(2*q2)*Bref_norm[1]-(2*q1)*Bref_norm[2], (2*q1)*Bref_norm[0]-(2*q0)*Bref_norm[1]+(2*q3)*Bref_norm[2]-(q3/q1)*((2*q3)*Bref_norm[0]+(2*q2)*Bref_norm[1]-(2*q1)*Bref_norm[2]), 0, 0, 0])
    
    H_mat = np.array([H1,H2,H3])

    return H_mat #JACOBIANO DEL MODELO DEL SENSOR (POR AHORA SOLO MAGNETOMETRO)

#%% EKF para el simulador si q0_k = qc
def Fq0(q0_k, q1_k, q2_k, q3_k, w0_k, w1_k, w2_k):
    
    F1 = np.array([0-(q1_k/q0_k)*(-0.5*w2_k), 0.5*w0_k-(q2_k/q0_k)*(-0.5*w2_k), 0.5*w1_k-(q3_k/q0_k)*(-0.5*w2_k), 0.5*q2_k, 0.5*q3_k, -0.5*q0_k])

    F2 = np.array([-0.5*w0_k-(q1_k/q0_k)*(0.5*w1_k), 0-(q2_k/q0_k)*(0.5*w1_k), 0.5*w2_k-(q3_k/q0_k)*(0.5*w1_k), -0.5*q1_k, 0.5*q0_k, 0.5*q3_k])

    F3 = np.array([-0.5*w1_k-(q1_k/q0_k)*(-0.5*w0_k), -0.5*w2_k-(q2_k/q0_k)*(-0.5*w0_k), 0-(q3_k/q0_k)*(-0.5*w0_k), -0.5*q0_k, -0.5*q1_k, -0.5*q2_k])

    F4 = np.array([0, 0, 0, 0, w2_k*(I_x-I_z)/I_y + tau/I_y, w1_k*(I_x-I_y)/I_z + tau/I_z])

    F5 = np.array([0, 0, 0, w2_k*(I_y-I_z)/I_x + tau/I_x, 0, w0_k*(I_y-I_z)/I_x + tau/I_x])

    F6 = np.array([0, 0, 0, w1_k*(I_x-I_y)/I_z + tau/I_z, w0_k*(I_x-I_y)/I_z + tau/I_z,0])

    F_k = np.array([F1,F2,F3,F4,F5,F6]) #JACOBIANO DEL VECTOR ESTADO
    #phi_k = np.exp(F_k) #PAPER ME DICE QUE LO DEJE ASI
    print("este es de q0")
    
    return F_k

def h_Xq0(q0_k, q1_k, q2_k, q3_k, w0_k, w1_k, w2_k,Bref):
    
    Bref_norm = Bref/np.linalg.norm(Bref)
    Bx_mod = (q3_k**2+q0_k**2-q1_k**2-q2_k**2)*Bref_norm[0]+ 2*(q0_k*q1_k+q2_k*q3_k)*Bref_norm[1] + 2*(q0_k*q2_k+q1_k*q3_k)*Bref_norm[2]
    By_mod = 2*(q0_k*q1_k-q2_k*q3_k)*Bref_norm[0] + (q3_k**2-q0_k**2+q1_k**2-q2_k**2)*Bref_norm[1] + 2*(q1_k*q2_k+q0_k*q3_k)*Bref_norm[2]
    Bz_mod = 2*(q0_k*q2_k+q1_k*q3_k)*Bref_norm[0] + 2*(q1_k*q2_k-q0_k*q3_k)*Bref_norm[1] + (q3_k**2-q0_k**2-q1_k**2+q2_k**2)*Bref_norm[2]
    
    B_mod = np.array([Bx_mod,By_mod,Bz_mod])
    return B_mod #entrega el z_k_priori

def Hq0(q0, q1, q2, q3, w0, w1, w2, Bref):  

    Bref_norm = Bref/np.linalg.norm(Bref)
    H1 = np.array([-(2*q1)*Bref_norm[0]+(2*q0)*Bref_norm[1]-(2*q3)*Bref_norm[2]-(q1/q0)*((2*q0)*Bref_norm[0]+(2*q1)*Bref_norm[1]+(2*q2)*Bref_norm[2]), -(2*q2)*Bref_norm[0]+(2*q3)*Bref_norm[1]+(2*q0)*Bref_norm[2]-(q2/q0)*((2*q0)*Bref_norm[0]+(2*q1)*Bref_norm[1]+(2*q2)*Bref_norm[2]), (2*q3)*Bref_norm[0]+(2*q2)*Bref_norm[1]-(2*q1)*Bref_norm[2]-(q3/q0)*((2*q0)*Bref_norm[0]+(2*q1)*Bref_norm[1]+(2*q2)*Bref_norm[2]), 0, 0, 0])
    H2 = np.array([(2*q0)*Bref_norm[0]+(2*q1)*Bref_norm[1]+(2*q2)*Bref_norm[2]-(q1/q0)*((2*q1)*Bref_norm[0]-(2*q0)*Bref_norm[1]+(2*q3)*Bref_norm[2]), -(2*q3)*Bref_norm[0]-(2*q2)*Bref_norm[1]+(2*q1)*Bref_norm[2]-(q2/q0)*((2*q1)*Bref_norm[0]-(2*q0)*Bref_norm[1]+(2*q3)*Bref_norm[2]), -(2*q2)*Bref_norm[0]+(2*q3)*Bref_norm[1]+(2*q0)*Bref_norm[2]-(q3/q0)*((2*q1)*Bref_norm[0]-(2*q0)*Bref_norm[1]+(2*q3)*Bref_norm[2]), 0, 0, 0])
    H3 = np.array([(2*q3)*Bref_norm[0]+(2*q2)*Bref_norm[1]-(2*q1)*Bref_norm[2]-(q1/q0)*((2*q2)*Bref_norm[0]-(2*q3)*Bref_norm[1]-(2*q0)*Bref_norm[2]), (2*q0)*Bref_norm[0]+(2*q1)*Bref_norm[1]+(2*q2)*Bref_norm[2]-(q2/q0)*((2*q2)*Bref_norm[0]-(2*q3)*Bref_norm[1]-(2*q0)*Bref_norm[2]), (2*q1)*Bref_norm[0]-(2*q0)*Bref_norm[1]+(2*q3)*Bref_norm[2]-(q3/q0)*((2*q2)*Bref_norm[0]-(2*q3)*Bref_norm[1]-(2*q0)*Bref_norm[2]), 0, 0, 0])
    
    H_mat = np.array([H1,H2,H3])

    return H_mat #JACOBIANO DEL MODELO DEL SENSOR (POR AHORA SOLO MAGNETOMETRO)


#%% demas matrices para el filtro
def Q(noise_mag, noise_gyros):
    Q1 = np.array([noise_mag**2,0,0,0,0,0])
    Q2 = np.array([0,noise_mag**2,0,0,0,0])
    Q3 = np.array([0,0,noise_mag**2,0,0,0])
    Q4 = np.array([0,0,0,noise_gyros**2,0,0])
    Q5 = np.array([0,0,0,0,noise_gyros**2,0])
    Q6 = np.array([0,0,0,0,0,noise_gyros**2])

    Q_k = np.array([Q1,Q2,Q3,Q4,Q5,Q6]) #MATRIZ DE COVARIANZA DEL RUIDO DE SENSORES
    
    return Q_k


def P_k_prior(phi_k, P_ki, Q_k):
    P_k_priori = np.dot(np.dot(phi_k,P_ki),np.transpose(phi_k)) + np.dot(phi_k,Q_k)
    return P_k_priori #MATRIZ P PRIORI CON P INICIAL DADO, DESPUES SE ACTUALIZA


def k_kalman(P_k_priori, H_mat):
    K_k_izq =  np.dot(P_k_priori,np.transpose(H_mat))
    K_k_der = np.linalg.inv(np.dot(np.dot(H_mat,P_k_priori),np.transpose(H_mat)))
    
    K_k = np.dot(K_k_izq,K_k_der)
    return K_k #GANANCIA DE KALMAN

def Delta_Xk(K_k, B_mod, B_medx, B_medy, B_medz):
    nu_x = B_medx-B_mod[0]
    nu_y = B_medy-B_mod[1]
    nu_z = B_medz-B_mod[2]
    nu_k = np.array([nu_x,nu_y,nu_z])
    Dxk = np.dot(K_k,nu_k)
    return Dxk #DELTA XK PARA OBTENER EL q y w POSTERIORI


def P_posteriori(K_k,H_k,P_k_priori):
   I = np.identity(6)
   P_k_pos = np.dot(I - np.dot(K_k,H_k),P_k_priori)
   return P_k_pos #SACAR MATRIZ P POSTERIORI ACTUALIZADA



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
    propagation_time = 340

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
    t_aux = []

    
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
    
    q0b_values = [qi_b[0]]
    q1b_values = [qi_b[1]]
    q2b_values = [qi_b[2]]
    q3b_values = [qi_b[3]]
    
    w0b_values = [w0_0]
    w1b_values = [w1_0]
    w2b_values = [w2_0]
    
    q0s_values = [qi_s[0]]
    q1s_values = [qi_s[1]]
    q2s_values = [qi_s[2]]
    q3s_values = [qi_s[3]]
    
    w0s_values = [w0_0]
    w1s_values = [w1_0]
    w2s_values = [w2_0]
    
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
        
        # Solucion de rk4 para las rotaciones
        for i in range(int(1/h)):
            t, q0, q1, q2, q3, w0, w1, w2 = t_values[-1], q0_values[-1], q1_values[-1], q2_values[-1], q3_values[-1], w0_values[-1], w1_values[-1], w2_values[-1]
            qn_new, w_new = rk4_step(t, q0, q1, q2, q3, w0, w1, w2, h)
            RPY =quaternion_to_euler(qn_new)
            
            t, q0_b, q1_b, q2_b, q3_b, w0_b, w1_b, w2_b = t_values[-1], q0b_values[-1], q1b_values[-1], q2b_values[-1], q3b_values[-1], w0b_values[-1], w1b_values[-1], w2b_values[-1]
            qb_new, wb_new = rk4_step(t, q0_b, q1_b, q2_b, q3_b, w0_b, w1_b, w2_b, h)
            
            t, q0_s, q1_s, q2_s, q3_s, w0_s, w1_s, w2_s = t_values[-1], q0s_values[-1], q1s_values[-1], q2s_values[-1], q3s_values[-1], w0s_values[-1], w1s_values[-1], w2s_values[-1]
            qs_new, ws_new = rk4_step(t, q0_s, q1_s, q2_s, q3_s, w0_s, w1_s, w2_s, h)
            
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
            
            q0b_values.append(qb_new[0])
            q1b_values.append(qb_new[1])
            q2b_values.append(qb_new[2])
            q3b_values.append(qb_new[3])
            w0b_values.append(wb_new[0])
            w1b_values.append(wb_new[1])
            w2b_values.append(wb_new[2])
            
            q0s_values.append(qs_new[0])
            q1s_values.append(qs_new[1])
            q2s_values.append(qs_new[2])
            q3s_values.append(qs_new[3])
            w0s_values.append(ws_new[0])
            w1s_values.append(ws_new[1])
            w2s_values.append(ws_new[2])
        
        tt = t_values[-1]
        
        #Para transformar el vector a LLA 
        lla = eci2lla(position,current_time_gps)
        
        #Obtener fuerzas magneticas de la Tierra
        B = pyIGRF.igrf_value(lla[0],lla[1],lla[2], current_time.year)
        B_f = np.array([B[3], B[4], B[5]])
        
        #Fuerza magnetica en bodyframe
        q_b = np.array([q0b_values[-1],q1b_values[-1],q2b_values[-1],q3b_values[-1]]) #cuaternion activo q = [cos(45/2), sin(45/2)*1/raiz(3),sin(45/2)*1/raiz(3),sin(45/2)*1/raiz(3)] normalizado
        B_eci = [0,B_f[0],B_f[1],B_f[2]]
        inv_q_b = inv_q(q_b)
        B_body = q_b*B_eci*inv_q_b
        B_body = np.array([B_body[1],B_body[2],B_body[3]]) 
        
        #obtener vector sol ECI
        jd2000 = datetime_to_jd2000(current_time)
        #########print(jd2000)
        sunvector = sun_vector(jd2000)
        
        #vector sol en body
        q_s = np.array([q0s_values[-1],q1s_values[-1],q2s_values[-1],q3s_values[-1]]) #cuaternion activo q = [cos(45/2), sin(45/2)*1/raiz(3),sin(45/2)*1/raiz(3),sin(45/2)*1/raiz(3)] normalizado
        vsun_eci = [0,sunvector[0],sunvector[1],sunvector[2]]
        inv_q_s = inv_q(q_s)
        vsun_body = q_s*vsun_eci*inv_q_s
        vsun_body = np.array([vsun_body[1],vsun_body[2],vsun_body[3]]) 

        
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
        t_aux.append(tt)

        
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
    RPY_values = np.array(RPY_values)
    t_aux = np.array(t_aux)
    
    
    #%% Sensores con ruido
    
    #sun sensor malo con ruido

    x_sun_vector_bad = simulate_sunsensor_reading(vsun_bodys[:,0],sigma_bad)
    x_sun_vector_bad = np.array(x_sun_vector_bad)
    y_sun_vector_bad = simulate_sunsensor_reading(vsun_bodys[:,1],sigma_bad)
    y_sun_vector_bad = np.array(y_sun_vector_bad)
    z_sun_vector_bad = simulate_sunsensor_reading(vsun_bodys[:,2],sigma_bad)
    z_sun_vector_bad = np.array(z_sun_vector_bad)
    
    #sun sensor medio con ruido
    x_sun_vector_med = simulate_sunsensor_reading(vsun_bodys[:,0],sigma_med)
    x_sun_vector_med = np.array(x_sun_vector_med)
    y_sun_vector_med = simulate_sunsensor_reading(vsun_bodys[:,1],sigma_med)
    y_sun_vector_med = np.array(y_sun_vector_med)
    z_sun_vector_med = simulate_sunsensor_reading(vsun_bodys[:,2],sigma_med)
    z_sun_vector_med = np.array(z_sun_vector_med)
    
    #sun sensor bueno con ruido
    x_sun_vector_good = simulate_sunsensor_reading(vsun_bodys[:,0],sigma_good)
    x_sun_vector_good= np.array(x_sun_vector_good)
    y_sun_vector_good = simulate_sunsensor_reading(vsun_bodys[:,1],sigma_good)
    y_sun_vector_good= np.array(y_sun_vector_good)
    z_sun_vector_good = simulate_sunsensor_reading(vsun_bodys[:,2],sigma_good)
    z_sun_vector_good= np.array(z_sun_vector_good)
    
    
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

    #%% EKF
    
    q_k00 = [q[0]]
    q_k10 = [q[1]]
    q_k20 = [q[2]]
    q_k30 = [q[3]]
    w_k00 = [w0_0]
    w_k10 = [w1_0]
    w_k20 = [w2_0]
    
    P_ki = np.identity(6) #P para iniciar el kiltro de kalman

    
    #FILTRO DE KALMAN EXTENDIDO TRUNCADO BAD GYROS BAD MAGNETO
    for i in range(len(t_aux)-1):
        Xk = f(t_aux[i],q_k00[-1],q_k10[-1],q_k20[-1],q_k30[-1],w_k00[-1], w_k10[-1], w_k20[-1])
        Xk = np.array(Xk)
        q_k = [Xk[0],Xk[1],Xk[2],Xk[3]]
        w_k = [Xk[4],Xk[5],Xk[6]]
        
        if abs(q_k[3])>abs(q_k[2]) and abs(q_k[3])>abs(q_k[1]) and abs(q_k[3])>abs(q_k[0]):
            q_k[3] = np.sqrt(1-(q_k[0])**2-(q_k[1])**2-(q_k[2])**2)
            F_k = Fq3(float(q_k[0]),float(q_k[1]),float(q_k[2]),float(q_k[3]),float(w_k[0]),float(w_k[1]),float(w_k[2]))
            Q_k = Q(ruido_bad, noise_rms_bad)
            P_k_priori = P_k_prior(F_k, P_ki, Q_k)
            H_k = Hq3(float(q_k[0]),float(q_k[1]),float(q_k[2]),float(q_k[3]),float(w_k[0]),float(w_k[1]),float(w_k[2]), Bs[i,:]) #B_s IGRF
            K_k = k_kalman(P_k_priori, H_k)
            B_mod = h_Xq3(float(q_k[0]),float(q_k[1]),float(q_k[2]),float(q_k[3]),float(w_k[0]),float(w_k[1]),float(w_k[2]),Bs[i,:])
            d_Xk = Delta_Xk(K_k, B_mod, Bb_magn_x[i],Bb_magn_y[i],Bb_magn_z[i])
            d_q = np.array([d_Xk[0], d_Xk[1],d_Xk[2]])
            d_w = np.array([d_Xk[3],d_Xk[4],d_Xk[5]])
            q_k_pos = np.array([q_k[0]+d_q[0], q_k[1]+d_q[1], q_k[2]+d_q[2]])
            w_k_pos = np.array([w_k[0]+d_w[0], w_k[1]+d_w[1], w_k[2]+d_w[2]])
        
            q_1_pos = q_k_pos[0]/np.sqrt(q_k[3]**2+q_k_pos[0]**2+q_k_pos[1]**2+q_k_pos[2]**2)
            q_2_pos = q_k_pos[1]/np.sqrt(q_k[3]**2+q_k_pos[0]**2+q_k_pos[1]**2+q_k_pos[2]**2)
            q_3_pos = q_k_pos[2]/np.sqrt(q_k[3]**2+q_k_pos[0]**2+q_k_pos[1]**2+q_k_pos[2]**2)
            q_4_pos = q_k[3] / np.sqrt(q_k[3]**2+q_k_pos[0]**2+q_k_pos[1]**2+q_k_pos[2]**2)

            P_k_pos= P_posteriori(K_k,H_k,P_k_priori)
            
            q_k00.append(float(q_1_pos))
            q_k10.append(float(q_2_pos))
            q_k20.append(float(q_3_pos))
            q_k30.append(float(q_4_pos))
            w_k00.append(float(w_k_pos[0]))
            w_k10.append(float(w_k_pos[1]))
            w_k20.append(float(w_k_pos[2]))
            P_ki = P_k_pos
            
        elif abs(q_k[2])>abs(q_k[3]) and abs(q_k[2])>abs(q_k[1]) and abs(q_k[2])>abs(q_k[0]):
            q_k[2] = np.sqrt(1-(q_k[0])**2-(q_k[1])**2-(q_k[3])**2)
            F_k = Fq2(float(q_k[0]),float(q_k[1]),float(q_k[2]),float(q_k[3]),float(w_k[0]),float(w_k[1]),float(w_k[2]))
            Q_k = Q(ruido_bad, noise_rms_bad)
            P_k_priori = P_k_prior(F_k, P_ki, Q_k)
            H_k = Hq2(float(q_k[0]),float(q_k[1]),float(q_k[2]),float(q_k[3]),float(w_k[0]),float(w_k[1]),float(w_k[2]), Bs[i,:]) #B_s IGRF
            K_k = k_kalman(P_k_priori, H_k)
            B_mod = h_Xq2(float(q_k[0]),float(q_k[1]),float(q_k[2]),float(q_k[3]),float(w_k[0]),float(w_k[1]),float(w_k[2]),Bs[i,:])
            d_Xk = Delta_Xk(K_k, B_mod, Bb_magn_x[i],Bb_magn_y[i],Bb_magn_z[i])
            d_q = np.array([d_Xk[0], d_Xk[1],d_Xk[2]])
            d_w = np.array([d_Xk[3],d_Xk[4],d_Xk[5]])
            q_k_pos = np.array([q_k[0]+d_q[0], q_k[1]+d_q[1], q_k[2]+d_q[2]])
            w_k_pos = np.array([w_k[0]+d_w[0], w_k[1]+d_w[1], w_k[2]+d_w[2]])
        
            q_1_pos = q_k_pos[0]/np.sqrt(q_k[2]**2+q_k_pos[0]**2+q_k_pos[1]**2+q_k_pos[2]**2)
            q_2_pos = q_k_pos[1]/np.sqrt(q_k[2]**2+q_k_pos[0]**2+q_k_pos[1]**2+q_k_pos[2]**2)
            q_4_pos = q_k_pos[2]/np.sqrt(q_k[2]**2+q_k_pos[0]**2+q_k_pos[1]**2+q_k_pos[2]**2)
            q_3_pos = q_k[2] / np.sqrt(q_k[2]**2+q_k_pos[0]**2+q_k_pos[1]**2+q_k_pos[2]**2)
            P_k_pos= P_posteriori(K_k,H_k,P_k_priori)
            
            q_k00.append(float(q_1_pos))
            q_k10.append(float(q_2_pos))
            q_k20.append(float(q_3_pos))
            q_k30.append(float(q_4_pos))
            w_k00.append(float(w_k_pos[0]))
            w_k10.append(float(w_k_pos[1]))
            w_k20.append(float(w_k_pos[2]))
            P_ki = P_k_pos
            
        elif abs(q_k[1])>abs(q_k[3]) and abs(q_k[1])>abs(q_k[2]) and abs(q_k[1])>abs(q_k[0]):
            q_k[1] = np.sqrt(1-(q_k[0])**2-(q_k[2])**2-(q_k[3])**2)
            F_k = Fq1(float(q_k[0]),float(q_k[1]),float(q_k[2]),float(q_k[3]),float(w_k[0]),float(w_k[1]),float(w_k[2]))
            Q_k = Q(ruido_bad, noise_rms_bad)
            P_k_priori = P_k_prior(F_k, P_ki, Q_k)
            H_k = Hq1(float(q_k[0]),float(q_k[1]),float(q_k[2]),float(q_k[3]),float(w_k[0]),float(w_k[1]),float(w_k[2]), Bs[i,:]) #B_s IGRF
            K_k = k_kalman(P_k_priori, H_k)
            B_mod = h_Xq1(float(q_k[0]),float(q_k[1]),float(q_k[2]),float(q_k[3]),float(w_k[0]),float(w_k[1]),float(w_k[2]),Bs[i,:])
            d_Xk = Delta_Xk(K_k, B_mod, Bb_magn_x[i],Bb_magn_y[i],Bb_magn_z[i])
            d_q = np.array([d_Xk[0], d_Xk[1],d_Xk[2]])
            d_w = np.array([d_Xk[3],d_Xk[4],d_Xk[5]])
            q_k_pos = np.array([q_k[0]+d_q[0], q_k[1]+d_q[1], q_k[2]+d_q[2]])
            w_k_pos = np.array([w_k[0]+d_w[0], w_k[1]+d_w[1], w_k[2]+d_w[2]])
        
            q_1_pos = q_k_pos[0]/np.sqrt(q_k[1]**2+q_k_pos[0]**2+q_k_pos[1]**2+q_k_pos[2]**2)
            q_3_pos = q_k_pos[1]/np.sqrt(q_k[1]**2+q_k_pos[0]**2+q_k_pos[1]**2+q_k_pos[2]**2)
            q_4_pos = q_k_pos[2]/np.sqrt(q_k[1]**2+q_k_pos[0]**2+q_k_pos[1]**2+q_k_pos[2]**2)
            q_2_pos = q_k[1] / np.sqrt(q_k[1]**2+q_k_pos[0]**2+q_k_pos[1]**2+q_k_pos[2]**2)
            P_k_pos= P_posteriori(K_k,H_k,P_k_priori)
            
            q_k00.append(float(q_1_pos))
            q_k10.append(float(q_2_pos))
            q_k20.append(float(q_3_pos))
            q_k30.append(float(q_4_pos))
            w_k00.append(float(w_k_pos[0]))
            w_k10.append(float(w_k_pos[1]))
            w_k20.append(float(w_k_pos[2]))
            P_ki = P_k_pos
            
        elif abs(q_k[0])>abs(q_k[3]) and abs(q_k[0])>abs(q_k[1]) and abs(q_k[0])>abs(q_k[2]):
            q_k[0] = np.sqrt(1-(q_k[1])**2-(q_k[2])**2-(q_k[3])**2)
            F_k = Fq0(float(q_k[0]),float(q_k[1]),float(q_k[2]),float(q_k[3]),float(w_k[0]),float(w_k[1]),float(w_k[2]))
            Q_k = Q(ruido_bad, noise_rms_bad)
            P_k_priori = P_k_prior(F_k, P_ki, Q_k)
            H_k = Hq0(float(q_k[0]),float(q_k[1]),float(q_k[2]),float(q_k[3]),float(w_k[0]),float(w_k[1]),float(w_k[2]), Bs[i,:]) #B_s IGRF
            K_k = k_kalman(P_k_priori, H_k)
            B_mod = h_Xq0(float(q_k[0]),float(q_k[1]),float(q_k[2]),float(q_k[3]),float(w_k[0]),float(w_k[1]),float(w_k[2]),Bs[i,:])
            d_Xk = Delta_Xk(K_k, B_mod, Bb_magn_x[i],Bb_magn_y[i],Bb_magn_z[i])
            d_q = np.array([d_Xk[0], d_Xk[1],d_Xk[2]])
            d_w = np.array([d_Xk[3],d_Xk[4],d_Xk[5]])
            q_k_pos = np.array([q_k[0]+d_q[0], q_k[1]+d_q[1], q_k[2]+d_q[2]])
            w_k_pos = np.array([w_k[0]+d_w[0], w_k[1]+d_w[1], w_k[2]+d_w[2]])
        
            q_2_pos = q_k_pos[0]/np.sqrt(q_k[0]**2+q_k_pos[0]**2+q_k_pos[1]**2+q_k_pos[2]**2)
            q_3_pos = q_k_pos[1]/np.sqrt(q_k[0]**2+q_k_pos[0]**2+q_k_pos[1]**2+q_k_pos[2]**2)
            q_4_pos = q_k_pos[2]/np.sqrt(q_k[0]**2+q_k_pos[0]**2+q_k_pos[1]**2+q_k_pos[2]**2)
            q_1_pos = q_k[0] / np.sqrt(q_k[0]**2+q_k_pos[0]**2+q_k_pos[1]**2+q_k_pos[2]**2)
            P_k_pos= P_posteriori(K_k,H_k,P_k_priori)
            
            q_k00.append(float(q_1_pos))
            q_k10.append(float(q_2_pos))
            q_k20.append(float(q_3_pos))
            q_k30.append(float(q_4_pos))
            w_k00.append(float(w_k_pos[0]))
            w_k10.append(float(w_k_pos[1]))
            w_k20.append(float(w_k_pos[2]))
            P_ki = P_k_pos
    
    # q_k00g = [q[0]]
    # q_k10g = [q[1]]
    # q_k20g = [q[2]]
    # q_k30g = [q[3]]
    # w_k00g = [w0_0]
    # w_k10g = [w1_0]
    # w_k20g = [w2_0]
    
    
    # #FILTRO DE KALMAN EXTENDIDO GOOD GYROS GOOD MAGNETO
    # for i in range(len(t_aux)-1):
    #     Xk = f(t_aux[i],q_k00[-1],q_k10[-1],q_k20[-1],q_k30[-1],w_k00[-1], w_k10[-1], w_k20[-1])
    #     Xk = np.array(Xk)
    #     q_k = [Xk[0],Xk[1],Xk[2],Xk[3]]
    #     w_k = [Xk[4],Xk[5],Xk[6]]
        
    #     if abs(q_k[3])>abs(q_k[2]) and abs(q_k[3])>abs(q_k[1]) and abs(q_k[3])>abs(q_k[0]):
    #         q_k[3] = np.sqrt(1-(q_k[0])**2-(q_k[1])**2-(q_k[2])**2)
    #         F_k = Fq3(float(q_k[0]),float(q_k[1]),float(q_k[2]),float(q_k[3]),float(w_k[0]),float(w_k[1]),float(w_k[2]))
    #         Q_k = Q(ruido, noise_rms_good)
    #         P_k_priori = P_k_prior(F_k, P_ki, Q_k)
    #         H_k = Hq3(float(q_k[0]),float(q_k[1]),float(q_k[2]),float(q_k[3]),float(w_k[0]),float(w_k[1]),float(w_k[2]), Bs[i,:]) #B_s IGRF
    #         K_k = k_kalman(P_k_priori, H_k)
    #         B_mod = h_Xq3(float(q_k[0]),float(q_k[1]),float(q_k[2]),float(q_k[3]),float(w_k[0]),float(w_k[1]),float(w_k[2]),Bs[i,:])
    #         d_Xk = Delta_Xk(K_k, B_mod, Bs_magn_x[i],Bs_magn_y[i],Bs_magn_z[i])
    #         d_q = np.array([d_Xk[0], d_Xk[1],d_Xk[2]])
    #         d_w = np.array([d_Xk[3],d_Xk[4],d_Xk[5]])
    #         q_k_pos = np.array([q_k[0]+d_q[0], q_k[1]+d_q[1], q_k[2]+d_q[2]])
    #         w_k_pos = np.array([w_k[0]+d_w[0], w_k[1]+d_w[1], w_k[2]+d_w[2]])
        
    #         q_1_pos = q_k_pos[0]/np.sqrt(q_k[3]**2+q_k_pos[0]**2+q_k_pos[1]**2+q_k_pos[2]**2)
    #         q_2_pos = q_k_pos[1]/np.sqrt(q_k[3]**2+q_k_pos[0]**2+q_k_pos[1]**2+q_k_pos[2]**2)
    #         q_3_pos = q_k_pos[2]/np.sqrt(q_k[3]**2+q_k_pos[0]**2+q_k_pos[1]**2+q_k_pos[2]**2)
    #         q_4_pos = q_k[3] / np.sqrt(q_k[3]**2+q_k_pos[0]**2+q_k_pos[1]**2+q_k_pos[2]**2)

    #         P_k_pos= P_posteriori(K_k,H_k,P_k_priori)
            
    #         q_k00g.append(float(q_1_pos))
    #         q_k10g.append(float(q_2_pos))
    #         q_k20g.append(float(q_3_pos))
    #         q_k30g.append(float(q_4_pos))
    #         w_k00g.append(float(w_k_pos[0]))
    #         w_k10g.append(float(w_k_pos[1]))
    #         w_k20g.append(float(w_k_pos[2]))
    #         P_ki = P_k_pos
            
    #     elif abs(q_k[2])>abs(q_k[3]) and abs(q_k[2])>abs(q_k[1]) and abs(q_k[2])>abs(q_k[0]):
    #         q_k[2] = np.sqrt(1-(q_k[0])**2-(q_k[1])**2-(q_k[3])**2)
    #         F_k = Fq2(float(q_k[0]),float(q_k[1]),float(q_k[2]),float(q_k[3]),float(w_k[0]),float(w_k[1]),float(w_k[2]))
    #         Q_k = Q(ruido, noise_rms_good)
    #         P_k_priori = P_k_prior(F_k, P_ki, Q_k)
    #         H_k = Hq2(float(q_k[0]),float(q_k[1]),float(q_k[2]),float(q_k[3]),float(w_k[0]),float(w_k[1]),float(w_k[2]), Bs[i,:]) #B_s IGRF
    #         K_k = k_kalman(P_k_priori, H_k)
    #         B_mod = h_Xq2(float(q_k[0]),float(q_k[1]),float(q_k[2]),float(q_k[3]),float(w_k[0]),float(w_k[1]),float(w_k[2]),Bs[i,:])
    #         d_Xk = Delta_Xk(K_k, B_mod, Bs_magn_x[i],Bs_magn_y[i],Bs_magn_z[i])
    #         d_q = np.array([d_Xk[0], d_Xk[1],d_Xk[2]])
    #         d_w = np.array([d_Xk[3],d_Xk[4],d_Xk[5]])
    #         q_k_pos = np.array([q_k[0]+d_q[0], q_k[1]+d_q[1], q_k[2]+d_q[2]])
    #         w_k_pos = np.array([w_k[0]+d_w[0], w_k[1]+d_w[1], w_k[2]+d_w[2]])
        
    #         q_1_pos = q_k_pos[0]/np.sqrt(q_k[2]**2+q_k_pos[0]**2+q_k_pos[1]**2+q_k_pos[2]**2)
    #         q_2_pos = q_k_pos[1]/np.sqrt(q_k[2]**2+q_k_pos[0]**2+q_k_pos[1]**2+q_k_pos[2]**2)
    #         q_4_pos = q_k_pos[2]/np.sqrt(q_k[2]**2+q_k_pos[0]**2+q_k_pos[1]**2+q_k_pos[2]**2)
    #         q_3_pos = q_k[2] / np.sqrt(q_k[2]**2+q_k_pos[0]**2+q_k_pos[1]**2+q_k_pos[2]**2)
    #         P_k_pos= P_posteriori(K_k,H_k,P_k_priori)
            
    #         q_k00g.append(float(q_1_pos))
    #         q_k10g.append(float(q_2_pos))
    #         q_k20g.append(float(q_3_pos))
    #         q_k30g.append(float(q_4_pos))
    #         w_k00g.append(float(w_k_pos[0]))
    #         w_k10g.append(float(w_k_pos[1]))
    #         w_k20g.append(float(w_k_pos[2]))
    #         P_ki = P_k_pos
            
    #     elif abs(q_k[1])>abs(q_k[3]) and abs(q_k[1])>abs(q_k[2]) and abs(q_k[1])>abs(q_k[0]):
    #         q_k[1] = np.sqrt(1-(q_k[0])**2-(q_k[2])**2-(q_k[3])**2)
    #         F_k = Fq1(float(q_k[0]),float(q_k[1]),float(q_k[2]),float(q_k[3]),float(w_k[0]),float(w_k[1]),float(w_k[2]))
    #         Q_k = Q(ruido, noise_rms_good)
    #         P_k_priori = P_k_prior(F_k, P_ki, Q_k)
    #         H_k = Hq1(float(q_k[0]),float(q_k[1]),float(q_k[2]),float(q_k[3]),float(w_k[0]),float(w_k[1]),float(w_k[2]), Bs[i,:]) #B_s IGRF
    #         K_k = k_kalman(P_k_priori, H_k)
    #         B_mod = h_Xq1(float(q_k[0]),float(q_k[1]),float(q_k[2]),float(q_k[3]),float(w_k[0]),float(w_k[1]),float(w_k[2]),Bs[i,:])
    #         d_Xk = Delta_Xk(K_k, B_mod, Bs_magn_x[i],Bs_magn_y[i],Bs_magn_z[i])
    #         d_q = np.array([d_Xk[0], d_Xk[1],d_Xk[2]])
    #         d_w = np.array([d_Xk[3],d_Xk[4],d_Xk[5]])
    #         q_k_pos = np.array([q_k[0]+d_q[0], q_k[1]+d_q[1], q_k[2]+d_q[2]])
    #         w_k_pos = np.array([w_k[0]+d_w[0], w_k[1]+d_w[1], w_k[2]+d_w[2]])
        
    #         q_1_pos = q_k_pos[0]/np.sqrt(q_k[1]**2+q_k_pos[0]**2+q_k_pos[1]**2+q_k_pos[2]**2)
    #         q_3_pos = q_k_pos[1]/np.sqrt(q_k[1]**2+q_k_pos[0]**2+q_k_pos[1]**2+q_k_pos[2]**2)
    #         q_4_pos = q_k_pos[2]/np.sqrt(q_k[1]**2+q_k_pos[0]**2+q_k_pos[1]**2+q_k_pos[2]**2)
    #         q_2_pos = q_k[1] / np.sqrt(q_k[1]**2+q_k_pos[0]**2+q_k_pos[1]**2+q_k_pos[2]**2)
    #         P_k_pos= P_posteriori(K_k,H_k,P_k_priori)
            
    #         q_k00g.append(float(q_1_pos))
    #         q_k10g.append(float(q_2_pos))
    #         q_k20g.append(float(q_3_pos))
    #         q_k30g.append(float(q_4_pos))
    #         w_k00g.append(float(w_k_pos[0]))
    #         w_k10g.append(float(w_k_pos[1]))
    #         w_k20g.append(float(w_k_pos[2]))
    #         P_ki = P_k_pos
            
    #     elif abs(q_k[0])>abs(q_k[3]) and abs(q_k[0])>abs(q_k[1]) and abs(q_k[0])>abs(q_k[2]):
    #         q_k[0] = np.sqrt(1-(q_k[1])**2-(q_k[2])**2-(q_k[3])**2)
    #         F_k = Fq0(float(q_k[0]),float(q_k[1]),float(q_k[2]),float(q_k[3]),float(w_k[0]),float(w_k[1]),float(w_k[2]))
    #         Q_k = Q(ruido, noise_rms_good)
    #         P_k_priori = P_k_prior(F_k, P_ki, Q_k)
    #         H_k = Hq0(float(q_k[0]),float(q_k[1]),float(q_k[2]),float(q_k[3]),float(w_k[0]),float(w_k[1]),float(w_k[2]), Bs[i,:]) #B_s IGRF
    #         K_k = k_kalman(P_k_priori, H_k)
    #         B_mod = h_Xq0(float(q_k[0]),float(q_k[1]),float(q_k[2]),float(q_k[3]),float(w_k[0]),float(w_k[1]),float(w_k[2]),Bs[i,:])
    #         d_Xk = Delta_Xk(K_k, B_mod, Bs_magn_x[i],Bs_magn_y[i],Bs_magn_z[i])
    #         d_q = np.array([d_Xk[0], d_Xk[1],d_Xk[2]])
    #         d_w = np.array([d_Xk[3],d_Xk[4],d_Xk[5]])
    #         q_k_pos = np.array([q_k[0]+d_q[0], q_k[1]+d_q[1], q_k[2]+d_q[2]])
    #         w_k_pos = np.array([w_k[0]+d_w[0], w_k[1]+d_w[1], w_k[2]+d_w[2]])
        
    #         q_2_pos = q_k_pos[0]/np.sqrt(q_k[0]**2+q_k_pos[0]**2+q_k_pos[1]**2+q_k_pos[2]**2)
    #         q_3_pos = q_k_pos[1]/np.sqrt(q_k[0]**2+q_k_pos[0]**2+q_k_pos[1]**2+q_k_pos[2]**2)
    #         q_4_pos = q_k_pos[2]/np.sqrt(q_k[0]**2+q_k_pos[0]**2+q_k_pos[1]**2+q_k_pos[2]**2)
    #         q_1_pos = q_k[0] / np.sqrt(q_k[0]**2+q_k_pos[0]**2+q_k_pos[1]**2+q_k_pos[2]**2)
    #         P_k_pos= P_posteriori(K_k,H_k,P_k_priori)
            
    #         q_k00g.append(float(q_1_pos))
    #         q_k10g.append(float(q_2_pos))
    #         q_k20g.append(float(q_3_pos))
    #         q_k30g.append(float(q_4_pos))
    #         w_k00g.append(float(w_k_pos[0]))
    #         w_k10g.append(float(w_k_pos[1]))
    #         w_k20g.append(float(w_k_pos[2]))
    #         P_ki = P_k_pos
    
    
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
    
    # plt.figure(figsize=(12, 6))
    # plt.plot(times, lambdas_bodys, label='lambdas body en sun sensor')
    # plt.xlabel('Tiempo')
    # plt.ylabel('lambda [rad]')
    # plt.legend()
    # plt.title('lambdas en body a lo largo del tiempo')
    # plt.grid()
    # plt.show()
    
    # plt.figure(figsize=(12, 6))
    # plt.plot(times, epsilon_bodys, label='epsilon body en sun sensor')
    # plt.xlabel('Tiempo')
    # plt.ylabel('epsilon [rad]')
    # plt.legend()
    # plt.title('epsilon en body a lo largo del tiempo')
    # plt.grid()
    # plt.show()
    
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
 
    #graficas vectores de sol con sun sensor malo
    plt.figure(figsize=(12, 6))
    plt.plot(times, vsun_bodys[:,0], label='Componente x vector sol body:')
    plt.plot(times,x_sun_vector_bad,label='Componente x vector sol body ruido malo:')
    plt.xlabel('Tiempo')
    plt.ylabel('Vector sol')
    plt.legend()
    plt.title('vector sol a lo largo del tiempo')
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(times, vsun_bodys[:,1], label='Componente y vector sol body:')
    plt.plot(times,y_sun_vector_bad,label='Componente y vector sol body ruido malo:')
    plt.xlabel('Tiempo')
    plt.ylabel('Vector sol')
    plt.legend()
    plt.title('vector sol a lo largo del tiempo')
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(times, vsun_bodys[:,2], label='Componente Z vector sol body:')
    plt.plot(times,z_sun_vector_bad,label='Componente Z vector sol body ruido malo:')
    plt.xlabel('Tiempo')
    plt.ylabel('Vector sol')
    plt.legend()
    plt.title('vector sol a lo largo del tiempo')
    plt.grid()
    plt.show()
    
    #graficas vectores de sol con sun sensor medio
    plt.figure(figsize=(12, 6))
    plt.plot(times, vsun_bodys[:,0], label='Componente x vector sol body:')
    plt.plot(times,x_sun_vector_med,label='Componente x vector sol body ruido medio:')
    plt.xlabel('Tiempo')
    plt.ylabel('Vector sol')
    plt.legend()
    plt.title('vector sol a lo largo del tiempo')
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(times, vsun_bodys[:,1], label='Componente y vector sol body:')
    plt.plot(times,y_sun_vector_med,label='Componente y vector sol body ruido medio:')
    plt.xlabel('Tiempo')
    plt.ylabel('Vector sol')
    plt.legend()
    plt.title('vector sol a lo largo del tiempo')
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(times, vsun_bodys[:,2], label='Componente Z vector sol body:')
    plt.plot(times,z_sun_vector_med,label='Componente Z vector sol body ruido medio:')
    plt.xlabel('Tiempo')
    plt.ylabel('Vector sol')
    plt.legend()
    plt.title('vector sol a lo largo del tiempo')
    plt.grid()
    plt.show()
    
    
    #graficas vectores de sol con sun sensor bueno
    plt.figure(figsize=(12, 6))
    plt.plot(times, vsun_bodys[:,0], label='Componente x vector sol body:')
    plt.plot(times,x_sun_vector_good,label='Componente x vector sol body ruido bueno:')
    plt.xlabel('Tiempo')
    plt.ylabel('Vector sol')
    plt.legend()
    plt.title('vector sol a lo largo del tiempo')
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(times, vsun_bodys[:,1], label='Componente y vector sol body:')
    plt.plot(times,y_sun_vector_good,label='Componente y vector sol body ruido bueno:')
    plt.xlabel('Tiempo')
    plt.ylabel('Vector sol')
    plt.legend()
    plt.title('vector sol a lo largo del tiempo')
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(times, vsun_bodys[:,2], label='Componente Z vector sol body:')
    plt.plot(times,z_sun_vector_good,label='Componente Z vector sol body ruido bueno:')
    plt.xlabel('Tiempo')
    plt.ylabel('Vector sol')
    plt.legend()
    plt.title('vector sol a lo largo del tiempo')
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(times, vsun_bodys[:,0], label='Componente x vector sol body:')
    plt.plot(times,x_sun_vector_bad,label='Componente x vector sol body ruido malo:')
    plt.plot(times,x_sun_vector_med,label='Componente x vector sol body ruido medio:')
    plt.plot(times,x_sun_vector_good,label='Componente x vector sol body ruido bueno:')
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
    
    #%% EKF
    
    # plt.figure(figsize=(12, 6))
    # plt.plot(t_aux, q_k00, label='q0 bad EKF')
    # plt.plot(t_aux, q_k10, label='q1 bad EKF')
    # plt.plot(t_aux, q_k20, label='q2 bad EKF')
    # plt.plot(t_aux, q_k30, label='q3 bad EKF')
    # plt.xlabel('Tiempo')
    # plt.ylabel('cuaternion EKF [-]')
    # plt.legend()
    # plt.title('Obtencion de los cuaterniones con EKF')
    # plt.grid()
    # plt.show()

    # plt.figure(figsize=(12, 6))
    # plt.plot(t_aux, w_k00, label='w0 bad EKF')
    # plt.plot(t_aux, w_k10, label='w1 bad EKF')
    # plt.plot(t_aux, w_k20, label='w2 bad EKF')
    # plt.xlabel('Tiempo')
    # plt.ylabel('velocidad angular EKF [rad/s]')
    # plt.legend()
    # plt.title('Obtencion de la velocidad angular con EKF')
    # plt.grid()
    # plt.show()
    
    # plt.figure(figsize=(12, 6))
    # plt.plot(t_aux, q_k00, label='q0 bad EKF')
    # plt.plot(t_values, q0_values, label='q0(t)')
    # plt.xlabel('Tiempo')
    # plt.ylabel('cuaternion EKF [-]')
    # plt.legend()
    # plt.title('Obtencion de los cuaterniones con EKF')
    # plt.grid()
    # plt.show()
    
    # #%% EKF GOOD
    
    # plt.figure(figsize=(12, 6))
    # plt.plot(t_aux, q_k00g, label='q0 good EKF')
    # plt.plot(t_aux, q_k10g, label='q1 good EKF')
    # plt.plot(t_aux, q_k20g, label='q2 good EKF')
    # plt.plot(t_aux, q_k30g, label='q3 good EKF')
    # plt.xlabel('Tiempo')
    # plt.ylabel('cuaternion EKF [-]')
    # plt.legend()
    # plt.title('Obtencion de los cuaterniones con EKF')
    # plt.grid()
    # plt.show()

    # plt.figure(figsize=(12, 6))
    # plt.plot(t_aux, w_k00g, label='w0 good EKF')
    # plt.plot(t_aux, w_k10g, label='w1 good EKF')
    # plt.plot(t_aux, w_k20g, label='w2 good EKF')
    # plt.xlabel('Tiempo')
    # plt.ylabel('velocidad angular EKF [rad/s]')
    # plt.legend()
    # plt.title('Obtencion de la velocidad angular con EKF')
    # plt.grid()
    # plt.show()
    
    # plt.figure(figsize=(12, 6))
    # plt.plot(t_aux, q_k00g, label='q0 good EKF')
    # plt.plot(t_values, q0_values, label='q0(t)')
    # plt.xlabel('Tiempo')
    # plt.ylabel('cuaternion EKF [-]')
    # plt.legend()
    # plt.title('Obtencion de los cuaterniones con EKF')
    # plt.grid()
    # plt.show()

