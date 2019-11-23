from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix
import numpy as np
from itertools import permutations
import matplotlib.pyplot as plt


def strategy_kmeans(data, k=3, seed=1):
    kmeans = KMeans(n_clusters=k, random_state=seed)
    kmeans.fit(data)
    centers = kmeans.cluster_centers_
    avg_largest_distance = round(distance_matrix(centers, centers).sum())

    solution = centers.copy()
    perms = list(permutations(np.arange(k)))

    solutions = (np.repeat([solution], len(perms), axis=0))
    distances = []
    for i, perm in enumerate(perms):
        largest_distance = avg_largest_distance
        for center in perm:
            centers = solutions[i].copy()
            for item in data[np.argwhere(kmeans.labels_ == center)]:
                centers[center] = item
                suggested_distance = round(distance_matrix(centers, centers).sum())
                if largest_distance < suggested_distance:
                    largest_distance = suggested_distance
                    solutions[i][center] = item
        distances.append(distance_matrix(solutions[i], solutions[i]).sum())
    distances = np.array(distances)
    solutions = solutions[np.argmax(distances)]
    return solutions
