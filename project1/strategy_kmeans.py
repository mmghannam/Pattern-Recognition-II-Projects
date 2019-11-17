from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix

def strategy_kmeans(data, k=3, seed=1):
    kmeans = KMeans(n_clusters=k, random_state=seed)
    kmeans.fit(data)
    kmeans.labels_
    centers = kmeans.cluster_centers_
    print(centers)
    print(distance_matrix(centers,centers))