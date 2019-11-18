import project1.strategy_kmeans as sk
import project1.Data as d
import numpy as np
from scipy.spatial import distance_matrix

if __name__ == '__main__':
    data = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    truth, data = d.Data(k=3, dim=3, n=1000)
    predictions = (sk.strategy_kmeans(data))
    print(distance_matrix(predictions, predictions).sum())
    print(distance_matrix(truth, truth).sum())

