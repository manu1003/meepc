from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
class Kmean:
    def __init__(self) -> None:
        pass

    def fit(self,X):
        cluster_range = range(2, min(11,len(X)))

        # Create empty lists to store the silhouette scores for each cluster number
        silhouette_scores = []

        # Iterate over the range of cluster numbers and fit KMeans++ models
        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters, init='k-means++')
            kmeans.fit(X)
            
            # Calculate the silhouette score for this model
            score = silhouette_score(X, kmeans.labels_)
            silhouette_scores.append(score)

        # Find the optimal number of clusters based on the highest silhouette score
        optimal_cluster_num = np.argmax(silhouette_scores) + 2

        return optimal_cluster_num