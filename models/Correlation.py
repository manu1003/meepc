import numpy as np
class Correlation:
    def __init__(self) -> None:
        pass
    def fit(self,X):
        means = X.mean(axis=0)
        M = X-means
        S = np.matmul(M.T,M)
        l = np.linalg.norm(M,axis=0)
        l += 1e-3
        for i in range(len(S)):
            for j in range(len(S)):
                S[i][j]/=l[i]
                S[i][j]/=l[j]
        return S