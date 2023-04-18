from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
class Cluster:
    def __init__(self):
        pass

    def fit(self, X, score='silhouette'):
        if len(np.unique(X)) == 1:
            kmeans = KMeans(n_clusters=1, init='k-means++')
            return kmeans.fit(X), 1

        cluster_range = range(1, min(10, len(np.unique(X))))

        # Create empty lists to store the score values and KMeans objects for each cluster number
        score_vals = []
        kmeans_objects = []

        # Iterate over the range of cluster numbers and fit KMeans++ models
        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters, init='k-means++')
            kmeans.fit(X)
            if n_clusters > 1:
                # Calculate the score for this model and store it along with the KMeans object
                if score == 'silhouette':
                    score_val = silhouette_score(X, kmeans.labels_)
                elif score == 'inertia':
                    score_val = kmeans.inertia_
                else:
                    raise ValueError('Invalid score parameter. Choose from "silhouette", or "inertia"')

                score_vals.append(score_val)
                kmeans_objects.append(kmeans)

        if score_vals:
            # Find the optimal number of clusters based on the highest score value (for silhouette and fmeasure) or lowest score value (for inertia)
            if score == 'silhouette':
                optimal_cluster_num = np.argmax(score_vals) + 1
            elif score == 'inertia':
                optimal_cluster_num = np.argmin(score_vals) + 1
            optimal_kmeans = kmeans_objects[optimal_cluster_num - 1]
            return optimal_kmeans, optimal_cluster_num
        else:
            return kmeans, 1
