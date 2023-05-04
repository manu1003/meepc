import numpy as np
from models import MEEPC

class Robustmeepc:

    def __init__(self) -> None:
        self.meepc=MEEPC()
        pass

    def fit(self,X,alpha_factor,beta_factor=.1):
        n,d = X.shape
        alpha = int(alpha_factor*n)
        beta = round(beta_factor*alpha)
        max_iter = round(1/beta_factor)
        for i in range(max_iter):
            weight,center = self.meepc.fit(X)
            #calculating radiis for all points
            var1 = np.square(X-center)
            radii = np.sqrt(np.matmul(weight,var1.T))
            idx = np.setdiff1d(np.arange(len(X)),np.argsort(radii)[-beta:])
            X = X[idx]
        weight,center = self.meepc.fit(X)
        return weight,center
            


