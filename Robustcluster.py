from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from copy import deepcopy
import numpy as np

class Robustcluster:
    def __init__(self) -> None:
        pass

    def fit(self,X,optimal_k,alpha_factor,y_train=None,max_iter=1000,tol=1e-4):
        alpha = round(alpha_factor*len(X))
        # Initialize centroids randomly
        if y_train is not None:
            safe_idx=np.where(y_train<1)[0]
            attack_idx=np.where(y_train>0)[0]
        centroid_old = X[np.random.choice(range(len(X)), optimal_k)]
        centroid_new = deepcopy(centroid_old)
        converged = False
        for _ in range(max_iter):
            # Assign data points to the nearest centroid
            distances = np.sqrt(np.sum((X - centroid_old[:,np.newaxis])**2,axis=2))
            idx = np.argsort(np.min(distances,axis=0))

            labels = np.argmin(distances,axis=0)

            labels_new = labels[idx[:-alpha]]
     
            X_new = X[idx[:-alpha]]
            #check for attack % in each itr

            common_elements = np.isin(original_array, subset_array)
            percentage = np.count_nonzero(common_elements) / len(original_array) * 100
            
            for i in range(optimal_k):
                centroid_new[i] = np.mean(X_new[labels_new==i],axis=0)

            if np.all(np.all(np.abs(centroid_new-centroid_old),axis=1) <= tol):
                break
            centroid_old = deepcopy(centroid_new)

        return centroid_new,labels







