from scipy.spatial import ConvexHull
from scipy.spatial import distance_matrix
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt


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
    for case in list(combinations(vertices, k - len(solution))):
        if len(solution) == 0:
            temp = data[list(case)]
        else:
            temp = np.concatenate((np.array(solution), data[list(case)]), axis=0)
        suggested_distance = round(distance_matrix(temp, temp).sum())
        print(temp)
        print(suggested_distance)
        if largest_distance < suggested_distance:
            largest_distance = suggested_distance
            best_addition = temp
    solution = best_addition
    return np.array(solution)






if __name__ == '__main__':
    print(strategy_convex_hull(data=np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]]), k=5))
