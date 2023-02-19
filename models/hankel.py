import numpy as np
class Hankel:
    def __init__(self):
        pass
        # self.X = X
        # self.lag = lag
        # self.stride = int(stride_percent*lag)

    def fit(self,X,lag,stride_percent):
        lag = lag*60
        stride = int(stride_percent*lag)
        hankel = X[:lag].reshape(-1,1)
        for i in range(stride,len(X),stride):
            if i+lag < len(X):
                new_col = X[i:i+lag].reshape(-1,1)
                hankel = np.concatenate((hankel,new_col),axis=1)
        return hankel