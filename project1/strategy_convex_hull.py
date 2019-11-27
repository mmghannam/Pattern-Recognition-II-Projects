from scipy.spatial import ConvexHull
from scipy.spatial import distance_matrix
import numpy as np
from itertools import combinations
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from scipy.spatial import distance


def strategy_convex_hull(data, k=4, seed=1):
    hull = ConvexHull(data)
    solution = []
    vertices = hull.vertices
    while len(solution) + len(vertices) <= k:
        solution.extend(data[vertices])
        if len(solution) == k:
            return np.array(solution)
        data = np.delete(data, vertices, axis=0)
        if data.shape[0] <= data.shape[1]:
            vertices = np.arange(data.shape[0])
            break
        hull = ConvexHull(data)
        vertices = hull.vertices
    largest_distance = 0.0
    max_distance = 0.0
    temp_index = vertices[0]
    temp = data[temp_index]
    solution.append(temp_index)
    vertices = vertices[vertices != temp_index]
    while (len(solution) < k):
        max_distance = 0.0
        for j in vertices:
            suggested_distance = round(distance.euclidean(temp, data[j]))
            if max_distance < suggested_distance:
                max_distance = suggested_distance
                best_addition = j
        solution.append(best_addition)
        temp = data[best_addition]
        vertices = vertices[vertices != best_addition]
    return np.array(solution)

