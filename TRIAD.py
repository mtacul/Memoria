# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 02:06:57 2023

@author: nachi
"""
import numpy as np

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

V1 = np.array([1,2,3])
V2 = np.array([4,5,6])
W1 = np.array([2,3,4])
W2 = np.array([3,4,5])

DCM = TRIAD(V1,V2,W1,W2)

print(DCM)