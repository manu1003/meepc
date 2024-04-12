from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from copy import deepcopy
import numpy as np

class Robustcluster:
    def __init__(self) -> None:
        pass

    def fit(self,X,optimal_k,alpha_factor,Labels=None,max_iter=10,tol=1e-10):
        alpha = round(alpha_factor*len(X))
        
        if Labels is not None:
            attack_idx=np.where(Labels>0)[0]

         # Initialize centroids randomly
        centroid_old = X[np.random.choice(range(len(X)), optimal_k)]
        centroid_new = deepcopy(centroid_old)
        converged = False


        # print("robust clustering iterations")
        for _ in range(max_iter):

            # Assign data points to the nearest centroid
            distances = np.sqrt(np.sum(X - centroid_old[:,np.newaxis],axis=2)**2)

            idx = np.argsort(np.min(distances,axis=0))

            labels = np.argmin(distances,axis=0)

            labels_new = labels[idx[:-alpha]]

            X_new = X[idx[:-alpha]]


            # check for attack % in each itr

            if Labels is not None and len(attack_idx) != 0:

                alpha_idx = idx[-alpha:]

                common_elements = np.intersect1d(attack_idx , alpha_idx)

                # print("recall in this iteration",len(common_elements)/len(attack_idx))


            for i in range(optimal_k):
                centroid_new[i] = np.mean(X_new[labels_new==i],axis=0)

            if np.all(np.all(np.abs(centroid_new-centroid_old),axis=1) <= tol):
                break
            centroid_old = deepcopy(centroid_new)
        if Labels is not None:
            print("------[CLUSTER] attack points found is {} ".format(len(common_elements)))

        return centroid_new,labels,alpha_idx







