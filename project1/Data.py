import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import os.path


def in_hull(p, hull):
    return hull.find_simplex(p) >= 0


def Data(k, dim, n):
    if os.path.exists("Data/data" + str(k) + "_" + str(dim) + "_" + str(n) + ".npy"):
        data = np.load("Data/data" + str(k) + "_" + str(dim) + "_" + str(n) + ".npy")
        ground_truth = data[0:k]
        shuffled_data = np.random.permutation(data)
        return ground_truth, shuffled_data
    data = []
    data_write = np.zeros((n, dim))
    radius = np.random.randint(0, dim * 10)
    squared_radius = radius * radius
    rad = np.full(k, radius)
    rad = np.square(rad)
    count = k
    i = 0
    while (i != k):
        point = np.random.uniform(-10 * dim, 10 * dim, (dim - 1))
        sum = np.sum(np.square(point))
        if (sum <= squared_radius):
            last_col = math.sqrt(squared_radius - sum)
            point = np.append(point, last_col)
            data.append(point)
            i = i + 1
    data = np.array(data)
    hull = Delaunay(data)
    for i in range(k):
        data_write[i] = data[i]
    while (count != n):
        point = np.random.uniform(-5 * dim, 5 * dim, (dim))
        val = in_hull(point, hull)
        if (val):
            data_write[count] = point
            count += 1
    np.save("Data/data" + str(k) + "_" + str(dim) + "_" + str(n), data_write)
    ground_truth = data
    shuffled_data = np.random.permutation(data_write)
    return ground_truth, shuffled_data
