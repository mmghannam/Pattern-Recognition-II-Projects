import numpy as np
import math
import os.path
from scipy.spatial import distance_matrix
from itertools import combinations

def Data(k,dim,n):
    if os.path.exists("Data/data" + str(k)+"_"+str(dim)+"_"+str(n)+ "_new.npy"):
        data = np.load("Data/data" + str(k)+"_"+str(dim)+"_"+str(n) + "_new.npy")
        ground_truth = data[0:k]
        shuffled_data = np.random.permutation(data[k:n+k])
        return ground_truth,shuffled_data
    data_write = np.zeros((n+k,dim))
    points = np.random.uniform(-10 * dim, 10 * dim, (n,dim))
    perm = combinations(points, k)
    max_distance = 0
    for i in perm:
        distance = distance_matrix(i,i).sum()
        if distance > max_distance:
            max_distance = distance
            ground_truth = i
    count =0
    for i in ground_truth:
        data_write[count] = i
        count=count+1
    data_write[k:n+k] = points
    np.save("Data/data" + str(k) + "_" + str(dim) + "_" + str(n)+"_new", data_write)
    shuffled_data = np.random.permutation(data_write[k:n+k])
    return np.array(ground_truth),shuffled_data

