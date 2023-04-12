import numpy as np
from models import Correlation
class Corrhankel:
    def __init__(self) -> None:
        self.corr = Correlation()
        pass

    def fit(self,X,lag,stride=0.5):
        # lag = lag*60
        stride = int(stride*lag)
        corr = self.corr.fit(X[:lag])
        no_of_lags=1
        for i in range(stride,len(X),stride):
            if i+lag < len(X):
                new_lag_corr = self.corr.fit(X[i:i+lag])
                corr = np.concatenate((corr,new_lag_corr),axis=0)
                no_of_lags+=1
        # corrhankel = np.concatenate((hankel,corr),axis = 0)
        return corr,no_of_lags
