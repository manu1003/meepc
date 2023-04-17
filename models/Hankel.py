import numpy as np
class Hankel:
    def __init__(self):
        pass
        # self.X = X
        # self.lag = lag
        # self.stride = int(stride_percent*lag)

    def fit(self,X,lag,stride_percent=0.5):
        # lag = lag*60
        stride = int(stride_percent*lag)
        hankel = X[:lag].reshape(-1,1)
        for i in range(stride,len(X),stride):
            if i+lag < len(X):
                new_col = X[i:i+lag].reshape(-1,1)
                hankel = np.concatenate((hankel,new_col),axis=1)
        min_col = np.min(hankel, axis=0)
        max_col = np.max(hankel, axis=0)
        mean_col = np.mean(hankel,axis=0)
        median_col = np.median(hankel, axis=0)
        hankel = np.append(hankel,[min_col],axis=0)
        hankel = np.append(hankel,[max_col],axis=0)
        hankel = np.append(hankel,[mean_col],axis=0)
        hankel = np.append(hankel,[median_col],axis=0)
        return hankel