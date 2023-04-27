from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

class Robustcluster:
    def __init__(self) -> None:
        pass

    import numpy as np

    def kmeans(X, K, max_iters=100):
        
        
        for i in range(max_iters):
            
            labels = np.argmin(distances, axis=0)
            
            # Update centroids to the mean of the assigned data points
            for k in range(K):
                centroids[k] = X[labels == k].mean(axis=0)
        
        return centroids, labels


    def fit(self,X,optimal_k,alpha):

        # Initialize centroids randomly
        centroid_old = X[np.random.choice(range(len(X)), optimal_k)]
        converged = False
        while not converged:
            # Assign data points to the nearest centroid
            for i in range(optimal_k):
                distances = np.sqrt(((X - centroid_old[i])**2).sum(axis=2))

        np.any(centroid_new-centroid_old) > .001:


