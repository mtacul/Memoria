# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 09:35:04 2023

@author: nachi
"""
I_x = 1
I_y = 0.5
I_z = 2
tau = 1e-5
import numpy as np
import matplotlib.pyplot as plt

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

#%%
# Paso 2: Define las funciones que describen las derivadas del sistema.
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
    #k2 = h * f1(x + 0.5 * h, y1 + 0.5 * k1, y2 + 0.5 * l1)
    
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

# Paso 4: Define las condiciones iniciales.
t0 = 0.0
q0_0 = 1.0
q1_0 = 0.0
q2_0 = 0.0
q3_0 = 0.0
w0_0 = 0
w1_0 = 0
w2_0 = 0

# Paso 5: Establece los valores de paso y el intervalo de integración.
h = 0.1
t_end = 60*60

# Paso 6: Ejecuta la integración numérica.
n_steps = int((t_end - t0) / h)
t_values = [t0]
q0_values = [q0_0]
q1_values = [q1_0]
q2_values = [q2_0]
q3_values = [q3_0]
w0_values = [w0_0]
w1_values = [w1_0]
w2_values = [w2_0]


for i in range(n_steps):
    t, q0, q1, q2, q3, w0, w1, w2 = t_values[-1], q0_values[-1], q1_values[-1], q2_values[-1], q3_values[-1], w0_values[-1], w1_values[-1], w2_values[-1]
    qn_new, w_new = rk4_step(t, q0, q1, q2, q3, w0, w1, w2, h)
    t_values.append(t + h)
    q0_values.append(qn_new[0])
    q1_values.append(qn_new[1])
    q2_values.append(qn_new[2])
    q3_values.append(qn_new[3])
    w0_values.append(w_new[0])
    w1_values.append(w_new[1])
    w2_values.append(w_new[2])

plt.plot(t_values, q0_values, label='q0(t)')
plt.plot(t_values, q1_values, label='q1(t)')
plt.plot(t_values, q2_values, label='q2(t)')
plt.plot(t_values, q3_values, label='q3(t)')
plt.xlabel('t [s]')
plt.ylabel('cuaternión')
plt.legend()
plt.show()

plt.plot(t_values, w0_values, label='w0(t)')
plt.plot(t_values, w1_values, label='w1(t)')
plt.plot(t_values, w2_values, label='w2(t)')
plt.xlabel('t [s]')
plt.ylabel('Vel. angular [rad/s]')
plt.legend()
plt.show()