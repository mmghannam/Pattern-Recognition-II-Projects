import project1.strategy_kmeans as sk
import project1.Data as d
import numpy as np
from scipy.spatial import distance_matrix
import timeit

if __name__ == '__main__':
    data = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

    k = 4
    truth, data = d.Data(k=k, dim=2, n=1000)
    predictions = (sk.strategy_kmeans(data, k=k))
    print(distance_matrix(predictions, predictions).sum())
    print(distance_matrix(truth, truth).sum())
