import numpy as np
import matplotlib.pyplot as plt
class Rank:
    def __init__(self) -> None:
        pass

    def fit(self,X,threshold=10):
        # Compute the eigenvalues and eigenvectors of the matrix
        eigenvalues= np.linalg.eigvals(np.matmul(X, X.T))

        if(len(np.unique(eigenvalues))==1):
            return min(X.shape[0],X.shape[1])-1

        sorted_eigenvalues = np.sort(eigenvalues)[::-1]

        if len(sorted_eigenvalues)>4 and np.any(sorted_eigenvalues)>0:
            thresh=0.1*sorted_eigenvalues[4]
            indices=np.where(sorted_eigenvalues>thresh)[0]
            r=len((list(indices)))
        else:
              r=len(sorted_eigenvalues)
        return r

