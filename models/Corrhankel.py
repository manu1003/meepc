import numpy as np
from models import Hankel,Correlation
class Corrhankel:
    def __init__(self) -> None:
        self.hankel = Hankel.Hankel()
        self.corr = Correlation.Correlation()
        pass

    def fit(self,X,sensor,lag,stride_percent=0.5):
        lag = lag*60
        stride = int(stride_percent*lag)
        hankel = self.hankel.fit(X[:,sensor],lag,stride_percent)
        corr = self.corr.fit(X[:lag],sensor).reshape(-1,1)
        for i in range(stride,len(X),stride):
            if i+lag < len(X):
                new_col = self.corr.fit(X[i:i+lag],sensor).reshape(-1,1)
                corr = np.concatenate((corr,new_col),axis=1)
        corrhankel = np.concatenate((hankel,corr),axis = 0)
        return corrhankel

