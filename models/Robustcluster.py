from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from copy import deepcopy
import numpy as np

class Robustcluster:
    def __init__(self) -> None:
        pass

    def fit(self,X,optimal_k,alpha_factor,tol=.001):
        alpha = round(alpha_factor*len(X))
        # Initialize centroids randomly
        centroid_old = X[np.random.choice(range(len(X)), optimal_k)]
        centroid_new = np.arange(optimal_k)
        converged = False
        while not converged:
            # Assign data points to the nearest centroid
            distances = np.sqrt(((X - centroid_old[:,np.newaxis])**2))
            idx = np.argsort(np.min(distances,axis=0))
            labels = np.argmin(distances,axis=0)
            labels_new = labels[idx[:-alpha]]
            X_new = X[idx[:-alpha]]
            for i in range(optimal_k):
                centroid_new[i] = np.mean(X_new[labels_new==i])

            if np.all(centroid_new-centroid_old) <= tol:
                converged = True
            centroid_old = deepcopy(centroid_new)

        return centroid_old,labels

            
            

        


