from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
class Cluster:
    def __init__(self) -> None:
        pass

    def fit(self,X):
        if len(np.unique(X)) == 1:
            kmeans = KMeans(n_clusters=1,init='k-means++')
            return kmeans.fit(X),1
        
        cluster_range = range(1, min(10,len(X)))
        
        # Create empty lists to store the silhouette scores and KMeans objects for each cluster number
        silhouette_scores = []
        kmeans_objects = []

        # Iterate over the range of cluster numbers and fit KMeans++ models
        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters, init='k-means++')
            kmeans.fit(X)
            if n_clusters>1:
                # Calculate the silhouette score for this model and store it along with the KMeans object
                score = silhouette_score(X, kmeans.labels_)
                silhouette_scores.append(score)
                kmeans_objects.append(kmeans)

        if silhouette_score:
            # Find the optimal number of clusters based on the highest silhouette score
            optimal_cluster_num = np.argmax(silhouette_scores) + 1 # Add 2 because the range starts at 2
            optimal_kmeans = kmeans_objects[optimal_cluster_num - 1] # Subtract 2 to get the index of the optimal number of clusters
            return optimal_kmeans,optimal_cluster_num
        else:
            return kmeans,1

        