from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from copy import deepcopy
import numpy as np

class Robustcluster:
    def __init__(self) -> None:
        pass

    def fit(self,X,optimal_k,alpha_factor,max_iter=1000,tol=1e-4):
        alpha = round(alpha_factor*len(X))
        # Initialize centroids randomly
        centroid_old = X[np.random.choice(range(len(X)), optimal_k)]
        # print(centroid_old,'\n')
        centroid_new = deepcopy(centroid_old)
        converged = False
        for _ in range(max_iter):
            # Assign data points to the nearest centroid
            distances = np.sqrt(np.sum((X - centroid_old[:,np.newaxis])**2,axis=2))
            # print(distances,'\n')
            idx = np.argsort(np.min(distances,axis=0))
            # print(idx,'\n')
            labels = np.argmin(distances,axis=0)
            # print(labels,'\n')
            labels_new = labels[idx[:-alpha]]
            # print(labels_new,'\n')
            X_new = X[idx[:-alpha]]
            # print(X_new,'\n')
            for i in range(optimal_k):
                centroid_new[i] = np.mean(X_new[labels_new==i],axis=0)

            if np.all(np.all(np.abs(centroid_new-centroid_old),axis=1) <= tol):
                break
            centroid_old = deepcopy(centroid_new)
        # print(centroid_new)
        return centroid_new,labels







