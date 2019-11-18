from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix
import numpy as np


def strategy_kmeans(data, k=3, seed=1):
    kmeans = KMeans(n_clusters=k, random_state=seed)
    kmeans.fit(data)
    centers = kmeans.cluster_centers_
    avg_largest_distance = distance_matrix(centers, centers).sum()

    solutions = centers.copy()
    for i in range(k):
        for item in data[np.argwhere(kmeans.labels_ == i)]:
            centers[i] = item
            suggested_distance = distance_matrix(centers, centers).sum()
            if avg_largest_distance < suggested_distance:
                avg_largest_distance = avg_largest_distance
                solutions[i] = item

    return solutions
