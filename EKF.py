# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 12:58:19 2023

@author: nachi
"""
import numpy as np
I_x = 1
I_y = 0.5
I_z = 2
tau = 1e-5 #torques externos de LEO SE DEBEN METER COMO ECUACIONES
noise_mag = 5
noise_gyros = 1
dt = 1

Bref = np.array([0.5,0.4,0.3])
B_med = np.array([0,-0.4,0])

q = np.array([0, 1, 0, 0]) #cuaternion activo q = [cos(45/2), sin(45/2)*1/raiz(3),sin(45/2)*1/raiz(3),sin(45/2)*1/raiz(3)] normalizado
w = np.array([0.33,0.33,0])

def f(dt, q0, q1, q2, q3, w0, w1, w2):
    q0_k = 0.5*(q1*w2 - q2*w1 + q3*w0)*dt
    q1_k = 0.5*(-q0*w2 + q2*w0 + q3*w1)*dt
    q2_k = 0.5*(q0*w1 - q1*w0 + q3*w2)*dt
    q3_k = 0.5*(-q0*w0 - q1*w1 - q2*w2)*dt
    w0_k = ((w1*w2*(I_y-I_z))/I_x + tau/I_x)*dt
    w1_k = ((w0*w2*(I_x-I_z))/I_y + tau/I_y)*dt
    w2_k = ((w0*w1*(I_x-I_y))/I_z + tau/I_z)*dt
    
    return q0_k,q1_k,q2_k,q3_k,w0_k,w1_k,w2_k #entrega el X_k

def h(q0_k, q1_k, q2_k, q3_k, w0_k, w1_k, w2_k,Bref):
    
    Bref_norm = Bref/np.linalg.norm(Bref)
    Bx_mod = (q3_k**2+q0_k**2-q1_k**2-q2_k**2)*Bref_norm[0]+ 2*(q0_k*q1_k+q2_k*q3_k)*Bref_norm[1] + 2*(q0_k*q2_k+q1_k*q3_k)*Bref_norm[2]
    By_mod = 2*(q0_k*q1_k-q2_k*q3_k)*Bref_norm[0] + (q3_k**2-q0_k**2+q1_k**2-q2_k**2)*Bref_norm[1] + 2*(q1_k*q2_k+q0_k*q3_k)*Bref_norm[2]
    Bz_mod = 2*(q0_k*q2_k+q1_k*q3_k)*Bref_norm[0] + 2*(q1_k*q2_k-q0_k*q3_k)*Bref_norm[1] + (q3_k**2-q0_k**2-q1_k**2+q2_k**2)*Bref_norm[2]
    
    B_mod = np.array([Bx_mod,By_mod,Bz_mod])
    
    return B_mod #entrega el z_k_priori

def F(q0_k, q1_k, q2_k, q3_k, w0_k, w1_k, w2_k):
    
    F1 = np.array([0, 0.5*w2_k, -0.5*w1_k, 0.5*w0_k, 0.5*q3_k, -0.5*q2_k, 0.5*q1_k])
    
    F2 = np.array([-0.5*w2_k, 0, 0.5*w0_k, 0.5*w1_k, 0.5*q2_k, 0.5*q3_k, -0.5*q0_k])
    
    F3 = np.array([0.5*w1_k, -0.5*w0_k, 0, 0.5*w2_k, -0.5*q1_k, 0.5*q0_k, 0.5*q3_k])
    
    F4 = np.array([-0.5*w0_k, -0.5*w1_k, -0.5*w2_k, 0, -0.5*q0_k, -0.5*q1_k, -0.5*q2_k])
    
    F5 = np.array([0, 0, 0, 0, 0, w2_k*(I_x-I_z)/I_y + tau/I_y, w1_k*(I_x-I_y)/I_z + tau/I_z])
    
    F6 = np.array([0, 0, 0, 0, w2_k*(I_y-I_z)/I_x + tau/I_x, 0, w0_k*(I_y-I_z)/I_x + tau/I_x])
    
    F7 = np.array([0, 0, 0, 0, w1_k*(I_x-I_y)/I_z + tau/I_z, w0_k*(I_x-I_y)/I_z + tau/I_z,0])
    
    F_k = np.array([F1,F2,F3,F4,F5,F6,F7])
    phi_k = np.exp(F_k)
    
    return phi_k


def Q(noise_mag, noise_gyros):
    Q1 = np.array([noise_mag**2,0,0,0,0,0,0])
    Q2 = np.array([0,noise_mag**2,0,0,0,0,0])
    Q3 = np.array([0,0,noise_mag**2,0,0,0,0])
    Q4 = np.array([0,0,0,noise_mag**2,0,0,0])
    Q5 = np.array([0,0,0,0,noise_gyros**2,0,0])
    Q6 = np.array([0,0,0,0,0,noise_gyros**2,0])
    Q7 = np.array([0,0,0,0,0,0,noise_gyros**2])

    Q_k = np.array([Q1,Q2,Q3,Q4,Q5,Q6,Q7])
    
    return Q_k


def P_k_prior(phi_k, P_ki, Q_k):
    P_k_priori = np.dot(np.dot(phi_k,P_ki),np.transpose(phi_k)) + np.dot(phi_k,Q_k)
    return P_k_priori


def H(q0, q1, q2, q3, w0, w1, w2, Bref):
    
    Bref_norm = Bref/np.linalg.norm(Bref)
    H1 = np.array([(2*q0)*Bref_norm[0]+(2*q1)*Bref_norm[1]+(2*q2)*Bref_norm[2], -(2*q1)*Bref_norm[0]+(2*q0)*Bref_norm[1]-(2*q3)*Bref_norm[2], -(2*q2)*Bref_norm[0]+(2*q3)*Bref_norm[1]+(2*q0)*Bref_norm[2], (2*q3)*Bref_norm[0]+(2*q2)*Bref_norm[1]-(2*q1)*Bref_norm[2], 0, 0, 0])
    H2 = np.array([(2*q1)*Bref_norm[0]-(2*q0)*Bref_norm[1]+(2*q3)*Bref_norm[2], (2*q0)*Bref_norm[0]+(2*q1)*Bref_norm[1]+(2*q2)*Bref_norm[2], -(2*q3)*Bref_norm[0]-(2*q2)*Bref_norm[1]+(2*q1)*Bref_norm[2], -(2*q2)*Bref_norm[0]+(2*q3)*Bref_norm[1]+(2*q0)*Bref_norm[2], 0, 0, 0])
    H3 = np.array([(2*q2)*Bref_norm[0]-(2*q3)*Bref_norm[1]-(2*q0)*Bref_norm[2], (2*q3)*Bref_norm[0]+(2*q2)*Bref_norm[1]-(2*q1)*Bref_norm[2], (2*q0)*Bref_norm[0]+(2*q1)*Bref_norm[1]+(2*q2)*Bref_norm[2], (2*q1)*Bref_norm[0]-(2*q0)*Bref_norm[1]+(2*q3)*Bref_norm[2], 0, 0, 0])
    
    H_mat = np.array([H1,H2,H3])
    return H_mat

def k_kalman(P_k_priori, H_mat):
    K_k_izq =  np.dot(P_k_priori,np.transpose(H_mat))
    K_k_der = np.linalg.inv(np.dot(np.dot(H_mat,P_k_priori),np.transpose(H_mat)))
    
    K_k = np.dot(K_k_izq,K_k_der)
    return K_k

def Delta_Xk(K_k, B_mod, B_med):
    nu_k = B_med-B_mod
    Dxk = np.dot(K_k,nu_k)
    return Dxk

def quat_posteriori(deltaq,q_k):
    deltaq_norm = deltaq/np.linalg.norm(deltaq)
    q_posteriori = np.zeros(4)
    q_posteriori[0] = q_k[0]*deltaq_norm[0] - q_k[1]*deltaq_norm[1] - q_k[2]*deltaq_norm[2] - q_k[3]*deltaq_norm[3]
    q_posteriori[1] = q_k[0]*deltaq_norm[1] + q_k[1]*deltaq_norm[0] + q_k[2]*deltaq_norm[3] - q_k[3]*deltaq_norm[2]
    q_posteriori[2] = q_k[0]*deltaq_norm[2] - q_k[1]*deltaq_norm[3] + q_k[2]*deltaq_norm[0] + q_k[3]*deltaq_norm[1]
    q_posteriori[3] = q_k[0]*deltaq_norm[3] + q_k[1]*deltaq_norm[2] - q_k[2]*deltaq_norm[1] + q_k[3]*deltaq_norm[0]
    return q_posteriori

def P_posteriori(K_k,H_k,P_k_priori):
   I = np.identity(7)
   P_k_pos = np.dot(I - np.dot(K_k,H_k),P_k_priori)
   return P_k_pos


Xk = f(dt,q[0],q[1],q[2],q[3],w[0],w[1],w[2])
q_k = np.array([Xk[0],Xk[1],Xk[2],Xk[3]])
w_k = np.array([Xk[4],Xk[5],Xk[6]])

phi_k = F(Xk[0],Xk[1],Xk[2],Xk[3],Xk[4],Xk[5],Xk[6])
Q_k = Q(noise_mag, noise_gyros)
P_ki = np.identity(7)
P_k_priori = P_k_prior(phi_k, P_ki, Q_k)

H_k = H(Xk[0],Xk[1],Xk[2],Xk[3],Xk[4],Xk[5],Xk[6], Bref)

K_k = k_kalman(P_k_priori, H_k)

B_mod = h(Xk[0],Xk[1],Xk[2],Xk[3],Xk[4],Xk[5],Xk[6],Bref)

d_Xk = Delta_Xk(K_k, B_mod, B_med)
d_q = np.array([d_Xk[0], d_Xk[1],d_Xk[2],d_Xk[3]])
d_w = np.array([d_Xk[4],d_Xk[5],d_Xk[6]])

q_k_pos = quat_posteriori(d_q,q_k)
w_k_pos = w_k + d_w
X_k_pos = np.array([q_k_pos[0],q_k_pos[1],q_k_pos[2], q_k_pos[3], w_k_pos[0], w_k_pos[1], w_k_pos[1]])
P_k_pos= P_posteriori(K_k,H_k,P_k_priori)

qq = [-0.0985418,
0.99508,
0.0102473,
-8.13197e-05]

qq_4 = np.sqrt(1-qq[0]**2-qq[1]**2-qq[2]**2)

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
def Fq3(q0_k, q1_k, q2_k, w0_k, w1_k, w2_k):
    q3_k = np.sqrt(1-q0_k**2-q1_k**2-q2_k**2)
    
    F1 = np.array([0-(q0_k/q3_k)*(0.5*w0_k), 0.5*w2_k-(q1_k/q3_k)*(0.5*w0_k), -0.5*w1_k-(q2_k/q3_k)*(0.5*w0_k), 0.5*q3_k, -0.5*q2_k, 0.5*q1_k])
    
    F2 = np.array([-0.5*w2_k-(q0_k/q3_k)*(0.5*w1_k), 0-(q1_k/q3_k)*(0.5*w1_k), 0.5*w0_k-(q2_k/q3_k)*(0.5*w1_k), 0.5*q2_k, 0.5*q3_k, -0.5*q0_k])
    
    F3 = np.array([0.5*w1_k-(q0_k/q3_k)*(0.5*w2_k), -0.5*w0_k-(q1_k/q3_k)*(0.5*w2_k), 0-(q2_k/q3_k)*(0.5*w2_k), -0.5*q1_k, 0.5*q0_k, 0.5*q3_k])
        
    F4 = np.array([0, 0, 0, 0, 0, w2_k*(I_x-I_z)/I_y + tau/I_y, w1_k*(I_x-I_y)/I_z + tau/I_z])
    
    F5 = np.array([0, 0, 0, 0, w2_k*(I_y-I_z)/I_x + tau/I_x, 0, w0_k*(I_y-I_z)/I_x + tau/I_x])
    
    F6 = np.array([0, 0, 0, 0, w1_k*(I_x-I_y)/I_z + tau/I_z, w0_k*(I_x-I_y)/I_z + tau/I_z,0])
    
    F_k = np.array([F1,F2,F3,F4,F5,F6]) #JACOBIANO DEL VECTOR ESTADO
    #phi_k = np.exp(F_k) #PAPER ME DICE QUE LO DEJE ASI
    
    return F_k

def h_Xq3(q0_k, q1_k, q2_k, w0_k, w1_k, w2_k,Bref):
    q3_k = np.sqrt(1-q0_k**2-q1_k**2-q2_k**2)
    
    Bref_norm = Bref/np.linalg.norm(Bref)
    Bx_mod = (q3_k**2+q0_k**2-q1_k**2-q2_k**2)*Bref_norm[0]+ 2*(q0_k*q1_k+q2_k*q3_k)*Bref_norm[1] + 2*(q0_k*q2_k+q1_k*q3_k)*Bref_norm[2]
    By_mod = 2*(q0_k*q1_k-q2_k*q3_k)*Bref_norm[0] + (q3_k**2-q0_k**2+q1_k**2-q2_k**2)*Bref_norm[1] + 2*(q1_k*q2_k+q0_k*q3_k)*Bref_norm[2]
    Bz_mod = 2*(q0_k*q2_k+q1_k*q3_k)*Bref_norm[0] + 2*(q1_k*q2_k-q0_k*q3_k)*Bref_norm[1] + (q3_k**2-q0_k**2-q1_k**2+q2_k**2)*Bref_norm[2]
    
    B_mod = np.array([Bx_mod,By_mod,Bz_mod])
    return B_mod #entrega el z_k_priori

def Hq3(q0, q1, q2, w0, w1, w2, Bref):  
    q3 = np.sqrt(1-q0**2-q1**2-q2**2)

    Bref_norm = Bref/np.linalg.norm(Bref)
    H1 = np.array([(2*q0)*Bref_norm[0]+(2*q1)*Bref_norm[1]+(2*q2)*Bref_norm[2]-(q0/q3)*((2*q3)*Bref_norm[0]+(2*q2)*Bref_norm[1]-(2*q1)*Bref_norm[2]), -(2*q1)*Bref_norm[0]+(2*q0)*Bref_norm[1]-(2*q3)*Bref_norm[2]-(q1/q3)*((2*q3)*Bref_norm[0]+(2*q2)*Bref_norm[1]-(2*q1)*Bref_norm[2]), -(2*q2)*Bref_norm[0]+(2*q3)*Bref_norm[1]+(2*q0)*Bref_norm[2]-(q2/q3)*((2*q3)*Bref_norm[0]+(2*q2)*Bref_norm[1]-(2*q1)*Bref_norm[2]), 0, 0, 0])
    H2 = np.array([(2*q1)*Bref_norm[0]-(2*q0)*Bref_norm[1]+(2*q3)*Bref_norm[2]-(q0/q3)*(-(2*q2)*Bref_norm[0]+(2*q3)*Bref_norm[1]+(2*q0)*Bref_norm[2]), (2*q0)*Bref_norm[0]+(2*q1)*Bref_norm[1]+(2*q2)*Bref_norm[2]-(q1/q3)*(-(2*q2)*Bref_norm[0]+(2*q3)*Bref_norm[1]+(2*q0)*Bref_norm[2]), -(2*q3)*Bref_norm[0]-(2*q2)*Bref_norm[1]+(2*q1)*Bref_norm[2]-(q2/q3)*(-(2*q2)*Bref_norm[0]+(2*q3)*Bref_norm[1]+(2*q0)*Bref_norm[2]), 0, 0, 0])
    H3 = np.array([(2*q2)*Bref_norm[0]-(2*q3)*Bref_norm[1]-(2*q0)*Bref_norm[2]-(q0/q3)*((2*q1)*Bref_norm[0]-(2*q0)*Bref_norm[1]+(2*q3)*Bref_norm[2]), (2*q3)*Bref_norm[0]+(2*q2)*Bref_norm[1]-(2*q1)*Bref_norm[2]-(q1/q3)*((2*q1)*Bref_norm[0]-(2*q0)*Bref_norm[1]+(2*q3)*Bref_norm[2]), (2*q0)*Bref_norm[0]+(2*q1)*Bref_norm[1]+(2*q2)*Bref_norm[2]-(q2/q3)*((2*q1)*Bref_norm[0]-(2*q0)*Bref_norm[1]+(2*q3)*Bref_norm[2]), 0, 0, 0])
    
    H_mat = np.array([H1,H2,H3])

    return H_mat #JACOBIANO DEL MODELO DEL SENSOR (POR AHORA SOLO MAGNETOMETRO)

#%% EKF para el simulador si q2_k = qc
def Fq2(q0_k, q1_k, q3_k, w0_k, w1_k, w2_k):
    q2_k = np.sqrt(1-q0_k**2-q1_k**2-q3_k**2)
    
    F1 = np.array([0-(q0_k/q2_k)*(-0.5*w1_k), 0.5*w2_k-(q1_k/q2_k)*(-0.5*w1_k), (0.5*w0_k)-(q3_k/q2_k)*(-0.5*w1_k), 0.5*q3_k, -0.5*q2_k, 0.5*q1_k])
    
    F2 = np.array([-0.5*w2_k-(q0_k/q2_k)*(0.5*w0_k), 0-(q1_k/q2_k)*((0.5*w0_k)), (0.5*w1_k)-(q3_k/q2_k)*(0.5*w0_k), 0.5*q2_k, 0.5*q3_k, -0.5*q0_k])
    
    F3 = np.array([-0.5*w0_k-(q0_k/q2_k)*(-0.5*w2_k), -0.5*w1_k-(q1_k/q2_k)*(-0.5*w2_k), 0-(q3_k/q2_k)*(-0.5*w1_k), -0.5*q0_k, -0.5*q1_k, -0.5*q2_k])
        
    F4 = np.array([0, 0, 0, 0, 0, w2_k*(I_x-I_z)/I_y + tau/I_y, w1_k*(I_x-I_y)/I_z + tau/I_z])
    
    F5 = np.array([0, 0, 0, 0, w2_k*(I_y-I_z)/I_x + tau/I_x, 0, w0_k*(I_y-I_z)/I_x + tau/I_x])
    
    F6 = np.array([0, 0, 0, 0, w1_k*(I_x-I_y)/I_z + tau/I_z, w0_k*(I_x-I_y)/I_z + tau/I_z,0])
    
    F_k = np.array([F1,F2,F3,F4,F5,F6]) #JACOBIANO DEL VECTOR ESTADO
    #phi_k = np.exp(F_k) #PAPER ME DICE QUE LO DEJE ASI
    
    return F_k

def h_Xq2(q0_k, q1_k, q3_k, w0_k, w1_k, w2_k,Bref):
    q2_k = np.sqrt(1-q0_k**2-q1_k**2-q3_k**2)
    
    Bref_norm = Bref/np.linalg.norm(Bref)
    Bx_mod = (q3_k**2+q0_k**2-q1_k**2-q2_k**2)*Bref_norm[0]+ 2*(q0_k*q1_k+q2_k*q3_k)*Bref_norm[1] + 2*(q0_k*q2_k+q1_k*q3_k)*Bref_norm[2]
    By_mod = 2*(q0_k*q1_k-q2_k*q3_k)*Bref_norm[0] + (q3_k**2-q0_k**2+q1_k**2-q2_k**2)*Bref_norm[1] + 2*(q1_k*q2_k+q0_k*q3_k)*Bref_norm[2]
    Bz_mod = 2*(q0_k*q2_k+q1_k*q3_k)*Bref_norm[0] + 2*(q1_k*q2_k-q0_k*q3_k)*Bref_norm[1] + (q3_k**2-q0_k**2-q1_k**2+q2_k**2)*Bref_norm[2]
    
    B_mod = np.array([Bx_mod,By_mod,Bz_mod])
    return B_mod #entrega el z_k_priori

def Hq2(q0, q1, q3, w0, w1, w2, Bref):  
    q2 = np.sqrt(1-q0**2-q1**2-q3**2)

    Bref_norm = Bref/np.linalg.norm(Bref)
    H1 = np.array([(2*q0)*Bref_norm[0]+(2*q1)*Bref_norm[1]+(2*q2)*Bref_norm[2]-(q0/q2)*-(2*q2)*Bref_norm[0]+(2*q3)*Bref_norm[1]+(2*q0)*Bref_norm[2],  -(2*q1)*Bref_norm[0]+(2*q0)*Bref_norm[1]-(2*q3)*Bref_norm[2]-(q1/q2)*-(2*q2)*Bref_norm[0]+(2*q3)*Bref_norm[1]+(2*q0)*Bref_norm[2], (2*q3)*Bref_norm[0]+(2*q2)*Bref_norm[1]-(2*q1)*Bref_norm[2]-(q3/q2)*-(2*q2)*Bref_norm[0]+(2*q3)*Bref_norm[1]+(2*q0)*Bref_norm[2], 0, 0, 0])
    H2 = np.array([(2*q1)*Bref_norm[0]-(2*q0)*Bref_norm[1]+(2*q3)*Bref_norm[2]-(q0/q2)*-(2*q3)*Bref_norm[0]-(2*q2)*Bref_norm[1]+(2*q1)*Bref_norm[2],  (2*q0)*Bref_norm[0]+(2*q1)*Bref_norm[1]+(2*q2)*Bref_norm[2]-(q1/q2)*-(2*q3)*Bref_norm[0]-(2*q2)*Bref_norm[1]+(2*q1)*Bref_norm[2], -(2*q2)*Bref_norm[0]+(2*q3)*Bref_norm[1]+(2*q0)*Bref_norm[2]-(q3/q2)*-(2*q3)*Bref_norm[0]-(2*q2)*Bref_norm[1]+(2*q1)*Bref_norm[2], 0, 0, 0])
    H3 = np.array([(2*q2)*Bref_norm[0]-(2*q3)*Bref_norm[1]-(2*q0)*Bref_norm[2]-(q0/q2)*(2*q0)*Bref_norm[0]+(2*q1)*Bref_norm[1]+(2*q2)*Bref_norm[2], (2*q3)*Bref_norm[0]+(2*q2)*Bref_norm[1]-(2*q1)*Bref_norm[2]-(q1/q2)*(2*q0)*Bref_norm[0]+(2*q1)*Bref_norm[1]+(2*q2)*Bref_norm[2],  (2*q1)*Bref_norm[0]-(2*q0)*Bref_norm[1]+(2*q3)*Bref_norm[2]-(q3/q2)*(2*q0)*Bref_norm[0]+(2*q1)*Bref_norm[1]+(2*q2)*Bref_norm[2], 0, 0, 0])
    
    
    H_mat = np.array([H1,H2,H3])

    return H_mat #JACOBIANO DEL MODELO DEL SENSOR (POR AHORA SOLO MAGNETOMETRO)

#%% EKF para el simulador si q1_k = qc
def Fq1(q0_k, q2_k, q3_k, w0_k, w1_k, w2_k):
    q1_k = np.sqrt(1-q0_k**2-q2_k**2-q3_k**2)
    
    F1 = np.array([0-(q0_k/q1_k)*0.5*w2_k, -0.5*w1_k-(q2_k/q1_k)*0.5*w2_k, 0.5*w0_k-(q3_k/q1_k)*0.5*w2_k, 0.5*q3_k, -0.5*q2_k, 0.5*q1_k])
        
    F2 = np.array([0.5*w1_k-(q0_k/q1_k)*(-0.5*w0_k), 0-(q2_k/q1_k)*(-0.5*w0_k), 0.5*w2_k-(q3_k/q1_k)*(-0.5*w0_k), -0.5*q1_k, 0.5*q0_k, 0.5*q3_k])

    F3 = np.array([-0.5*w0_k-(q0_k/q1_k)*(-0.5*w1_k), -0.5*w2_k-(q2_k/q1_k)*(-0.5*w1_k), 0-(q3_k/q1_k)*(-0.5*w1_k), -0.5*q0_k, -0.5*q1_k, -0.5*q2_k])
     
    F4 = np.array([0, 0, 0, 0, 0, w2_k*(I_x-I_z)/I_y + tau/I_y, w1_k*(I_x-I_y)/I_z + tau/I_z])
    
    F5 = np.array([0, 0, 0, 0, w2_k*(I_y-I_z)/I_x + tau/I_x, 0, w0_k*(I_y-I_z)/I_x + tau/I_x])
    
    F6 = np.array([0, 0, 0, 0, w1_k*(I_x-I_y)/I_z + tau/I_z, w0_k*(I_x-I_y)/I_z + tau/I_z,0])
    
    F_k = np.array([F1,F2,F3,F4,F5,F6]) #JACOBIANO DEL VECTOR ESTADO
    #phi_k = np.exp(F_k) #PAPER ME DICE QUE LO DEJE ASI
    
    return F_k

def h_Xq1(q0_k, q2_k, q3_k, w0_k, w1_k, w2_k,Bref):
    q1_k = np.sqrt(1-q0_k**2-q2_k**2-q3_k**2)
    
    Bref_norm = Bref/np.linalg.norm(Bref)
    Bx_mod = (q3_k**2+q0_k**2-q1_k**2-q2_k**2)*Bref_norm[0]+ 2*(q0_k*q1_k+q2_k*q3_k)*Bref_norm[1] + 2*(q0_k*q2_k+q1_k*q3_k)*Bref_norm[2]
    By_mod = 2*(q0_k*q1_k-q2_k*q3_k)*Bref_norm[0] + (q3_k**2-q0_k**2+q1_k**2-q2_k**2)*Bref_norm[1] + 2*(q1_k*q2_k+q0_k*q3_k)*Bref_norm[2]
    Bz_mod = 2*(q0_k*q2_k+q1_k*q3_k)*Bref_norm[0] + 2*(q1_k*q2_k-q0_k*q3_k)*Bref_norm[1] + (q3_k**2-q0_k**2-q1_k**2+q2_k**2)*Bref_norm[2]
    
    B_mod = np.array([Bx_mod,By_mod,Bz_mod])
    return B_mod #entrega el z_k_priori

def Hq1(q0, q2, q3, w0, w1, w2, Bref):  
    q1 = np.sqrt(1-q0**2-q2**2-q3**2)

    Bref_norm = Bref/np.linalg.norm(Bref)
    H1 = np.array([(2*q0)*Bref_norm[0]+(2*q1)*Bref_norm[1]+(2*q2)*Bref_norm[2]-(q0/q1)*(-(2*q1)*Bref_norm[0]+(2*q0)*Bref_norm[1]-(2*q3)*Bref_norm[2]), -(2*q2)*Bref_norm[0]+(2*q3)*Bref_norm[1]+(2*q0)*Bref_norm[2]-(q2/q1)*(-(2*q1)*Bref_norm[0]+(2*q0)*Bref_norm[1]-(2*q3)*Bref_norm[2]), (2*q3)*Bref_norm[0]+(2*q2)*Bref_norm[1]-(2*q1)*Bref_norm[2]-(q3/q1)*-(2*q1)*Bref_norm[0]+(2*q0)*Bref_norm[1]-(2*q3)*Bref_norm[2], 0, 0, 0])
    H2 = np.array([(2*q1)*Bref_norm[0]-(2*q0)*Bref_norm[1]+(2*q3)*Bref_norm[2]-(q0/q1)*((2*q0)*Bref_norm[0]+(2*q1)*Bref_norm[1]+(2*q2)*Bref_norm[2]), -(2*q3)*Bref_norm[0]-(2*q2)*Bref_norm[1]+(2*q1)*Bref_norm[2]-(q2/q1)*((2*q0)*Bref_norm[0]+(2*q1)*Bref_norm[1]+(2*q2)*Bref_norm[2]), -(2*q2)*Bref_norm[0]+(2*q3)*Bref_norm[1]+(2*q0)*Bref_norm[2]-(q3/q1)*((2*q0)*Bref_norm[0]+(2*q1)*Bref_norm[1]+(2*q2)*Bref_norm[2]), 0, 0, 0])
    H3 = np.array([(2*q2)*Bref_norm[0]-(2*q3)*Bref_norm[1]-(2*q0)*Bref_norm[2]-(q0/q1)*((2*q3)*Bref_norm[0]+(2*q2)*Bref_norm[1]-(2*q1)*Bref_norm[2]), (2*q0)*Bref_norm[0]+(2*q1)*Bref_norm[1]+(2*q2)*Bref_norm[2]-(q2/q1)*(2*q3)*Bref_norm[0]+(2*q2)*Bref_norm[1]-(2*q1)*Bref_norm[2], (2*q1)*Bref_norm[0]-(2*q0)*Bref_norm[1]+(2*q3)*Bref_norm[2]-(q3/q1)*((2*q3)*Bref_norm[0]+(2*q2)*Bref_norm[1]-(2*q1)*Bref_norm[2]), 0, 0, 0])
    
    H_mat = np.array([H1,H2,H3])

    return H_mat #JACOBIANO DEL MODELO DEL SENSOR (POR AHORA SOLO MAGNETOMETRO)

#%% EKF para el simulador si q0_k = qc
def Fq0(q1_k, q2_k, q3_k, w0_k, w1_k, w2_k):
    q0_k = np.sqrt(1-q1_k**2-q2_k**2-q3_k**2)
    
    F1 = np.array([0-(q1_k/q0_k)*(-0.5*w2_k), 0.5*w0_k-(q2_k/q0_k)*(-0.5*w2_k), 0.5*w1_k-(q3_k/q0_k)*(-0.5*w2_k), 0.5*q2_k, 0.5*q3_k, -0.5*q0_k])
    
    F2 = np.array([-0.5*w0_k-(q1_k/q0_k)*(0.5*w1_k), 0-(q2_k/q0_k)*(0.5*w1_k), 0.5*w2_k-(q3_k/q0_k)*(0.5*w1_k), -0.5*q1_k, 0.5*q0_k, 0.5*q3_k])
    
    F3 = np.array([-0.5*w1_k-(q1_k/q0_k)*(-0.5*w0_k), -0.5*w2_k-(q2_k/q0_k)*(-0.5*w0_k), 0-(q3_k/q0_k)*(-0.5*w0_k), -0.5*q0_k, -0.5*q1_k, -0.5*q2_k])
    
    F4 = np.array([0, 0, 0, 0, 0, w2_k*(I_x-I_z)/I_y + tau/I_y, w1_k*(I_x-I_y)/I_z + tau/I_z])
    
    F5 = np.array([0, 0, 0, 0, w2_k*(I_y-I_z)/I_x + tau/I_x, 0, w0_k*(I_y-I_z)/I_x + tau/I_x])
    
    F6 = np.array([0, 0, 0, 0, w1_k*(I_x-I_y)/I_z + tau/I_z, w0_k*(I_x-I_y)/I_z + tau/I_z,0])
    
    F_k = np.array([F1,F2,F3,F4,F5,F6]) #JACOBIANO DEL VECTOR ESTADO
    #phi_k = np.exp(F_k) #PAPER ME DICE QUE LO DEJE ASI
    
    return F_k

def h_Xq0(q1_k, q2_k, q3_k, w0_k, w1_k, w2_k,Bref):
    q0_k = np.sqrt(1-q1_k**2-q2_k**2-q3_k**2)
    
    Bref_norm = Bref/np.linalg.norm(Bref)
    Bx_mod = (q3_k**2+q0_k**2-q1_k**2-q2_k**2)*Bref_norm[0]+ 2*(q0_k*q1_k+q2_k*q3_k)*Bref_norm[1] + 2*(q0_k*q2_k+q1_k*q3_k)*Bref_norm[2]
    By_mod = 2*(q0_k*q1_k-q2_k*q3_k)*Bref_norm[0] + (q3_k**2-q0_k**2+q1_k**2-q2_k**2)*Bref_norm[1] + 2*(q1_k*q2_k+q0_k*q3_k)*Bref_norm[2]
    Bz_mod = 2*(q0_k*q2_k+q1_k*q3_k)*Bref_norm[0] + 2*(q1_k*q2_k-q0_k*q3_k)*Bref_norm[1] + (q3_k**2-q0_k**2-q1_k**2+q2_k**2)*Bref_norm[2]
    
    B_mod = np.array([Bx_mod,By_mod,Bz_mod])
    return B_mod #entrega el z_k_priori

def Hq0(q1, q2, q3, w0, w1, w2, Bref):  
    q0 = np.sqrt(1-q1**2-q2**2-q3**2)

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


#%% P posteriori para los 4 cuaterniones en caso que qc=q4

def P_posteriori_q4(P_posteriori,q3, q0, q1, q2):
    P_41 = -1/q3*(q0*P_posteriori[0,0]+q1*P_posteriori[1,0]+q2*P_posteriori[2,0])
    P_42 = -1/q3*(q0*P_posteriori[0,1]+q1*P_posteriori[1,1]+q2*P_posteriori[2,1])
    P_43 = -1/q3*(q0*P_posteriori[0,2]+q1*P_posteriori[1,2]+q2*P_posteriori[2,2])
    P_44 = 1/abs(q3)**2*()
    
#%%

def Fq1(q0_k, q1_k, q2_k, q3_k, w0_k, w1_k, w2_k):
    
    F1 = np.array([0-(q0_k/q1_k)*0.5*w2_k, -0.5*w1_k-(q2_k/q1_k)*0.5*w2_k, 0.5*w0_k-(q3_k/q1_k)*0.5*w2_k, 0.5*q3_k, -0.5*q2_k, 0.5*q1_k])
    print(F1.shape)    
    F2 = np.array([0.5*w1_k-(q0_k/q1_k)*(-0.5*w0_k), 0-(q2_k/q1_k)*(-0.5*w0_k), 0.5*w2_k-(q3_k/q1_k)*(-0.5*w0_k), -0.5*q1_k, 0.5*q0_k, 0.5*q3_k])
    print(F2.shape)    

    F3 = np.array([-0.5*w0_k-(q0_k/q1_k)*(-0.5*w1_k), -0.5*w2_k-(q2_k/q1_k)*(-0.5*w1_k), 0-(q3_k/q1_k)*(-0.5*w1_k), -0.5*q0_k, -0.5*q1_k, -0.5*q2_k])
    print(F3.shape)    

    F4 = np.array([0, 0, 0, 0, w2_k*(I_x-I_z)/I_y + tau/I_y, w1_k*(I_x-I_y)/I_z + tau/I_z])
    print(F4.shape)    

    F5 = np.array([0, 0, 0, w2_k*(I_y-I_z)/I_x + tau/I_x, 0, w0_k*(I_y-I_z)/I_x + tau/I_x])
    print(F5.shape)    

    F6 = np.array([0, 0, 0, w1_k*(I_x-I_y)/I_z + tau/I_z, w0_k*(I_x-I_y)/I_z + tau/I_z,0])
    print(F6.shape)    

    F_k = np.array([F1,F2,F3,F4,F5,F6]) #JACOBIANO DEL VECTOR ESTADO
    #phi_k = np.exp(F_k) #PAPER ME DICE QUE LO DEJE ASI
    
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

import numpy as np
I_x = 1
I_y = 0.5
I_z = 2
tau = 1e-5 #torques externos de LEO SE DEBEN METER COMO ECUACIONES
noise_mag = 5
noise_gyros = 1
dt = 1

Bref = np.array([0.5,0.4,0.3])
B_med = np.array([0,-0.4,0])

q = np.array([0, 1, 0, 0]) #cuaternion activo q = [cos(45/2), sin(45/2)*1/raiz(3),sin(45/2)*1/raiz(3),sin(45/2)*1/raiz(3)] normalizado
w = np.array([0.33,0.33,0])
q_k = [-0.0005164290612730166,
 0.9974969966701329,
 -0.054681098867781615,
 -0.0448269155943321]

w_k = [9.999999999999999e-06, 1.9999999999999998e-05, 0.002505]

F_k = Fq1(q_k[0],q_k[1],q_k[2],q_k[3],w_k[0],w_k[1],w_k[2])
