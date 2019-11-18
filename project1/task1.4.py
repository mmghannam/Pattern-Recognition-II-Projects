import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

def approxiamte_major_component(X_k):
    x_a, x_b = getDistantObjects(X_k)
    return (x_a - x_b) / la.norm(x_a - x_b)

def getDistantObjects(X_k):
    x_i = X_k[np.random.choice(X_k.shape[0])]
    x_j = X_k[np.argmax(np.apply_along_axis(la.norm, 0, (X_k - x_i)))]
    x_k = X_k[np.argmax(np.apply_along_axis(la.norm, 0, (X_k - x_j)))]
    x_l = X_k[np.argmax(np.apply_along_axis(la.norm, 0, (X_k - x_k)))]
    return x_k, x_l

if __name__ == '__main__':
    X = np.load('faceMatrix.npy').astype('float')
    X_k = X.transpose()
    v = []
    for i in range(5):
        v_k = approxiamte_major_component(X_k)
        v.append(v_k)
        X_k = np.dot(np.identity(X_k.shape[0]) - np.dot(v_k, v_k.transpose()) , X_k)