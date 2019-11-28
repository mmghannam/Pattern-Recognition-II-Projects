#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: shahnawaz
"""

import numpy as np
import numpy.linalg as la
from numpy.linalg import eigh
import timeit
import functools
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt


# Function to iteratively run QR evaluation
def C_QR(C, count):
    if count > 1:
        Q, R = la.qr(C[:, :, 10 - count])
        C[:, :, 11 - count] = np.matmul(R, Q)
        return C_QR(C, count - 1)
    else:
        Q, R = la.qr(C[:, :, 10 - count])
        C[:, :, 11 - count] = np.matmul(R, Q)
        return C


X = np.load('Data/faceMatrix.npy').astype('float')

X = normalize(X - np.mean(X, axis=0), axis=0)

C_0 = X.dot(np.transpose(X))

# Creating buffer for C accross iteration
C = np.zeros((np.shape(C_0)[0], np.shape(C_0)[1], 11))
C[:, :, 0] = C_0

# Backup C for time analysis
C_backup = C

# Evaluating QR Algorithm
C = C_QR(C, 10)

# Getting final C
C_10 = C[:, :, 10]

## Comparision
# eigh spectral decomposition
eigh_values, eigh_vectors = eigh(C_0)

# Eigen Values from spectral decomposition
eigen_values_Spectral = eigh_values[::-1]

# Eigen Values form QR Algorithm for the last computed C after 10 iteration
eigen_values_QR = C_10.diagonal()

# plot
number_of_values = len(eigen_values_QR)
plt.title('QR algorithm: diagonal(C_10)')
plt.bar(range(number_of_values), eigen_values_QR)
plt.savefig("Task 1-3")

# Evaluating error
error = eigen_values_Spectral - eigen_values_QR

print(np.mean(error))

# Average time analysis in secs
ts = timeit.Timer(functools.partial(C_QR, C_backup, 10)).repeat(3, 100)
print (min(ts) / 100)
