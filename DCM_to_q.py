# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 01:03:12 2023

@author: nachi
"""

import numpy as np
from scipy.spatial.transform import Rotation

# Define la matriz de rotación ECI a cuerpo (por ejemplo)
rotation_matrix_eci_to_body = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

# Convierte la matriz de rotación a un cuaternión
quaternion = Rotation.from_matrix(rotation_matrix_eci_to_body).as_quat()

# Imprime el cuaternión resultante
print("Cuaternión resultante:", quaternion)
