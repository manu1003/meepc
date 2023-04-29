import numpy as np
from models import MEEPC

class Robustmeepc:
    def __init__(self) -> None:
        self.meepc=MEEPC()
        pass
    def fit(self,X,alpha=0.05, max_iter=100, tol=1e-4):
        n,d =X.shape
        alpha=int(alpha*n)
        for i in range(max_iter):
            weight1,center1=self.meepc.fit(X)
            #calculating radiis for all points
            var1=np.square(X-center1)
            radiis=np.sqrt(np.matmul(weight1,var1.T))
            idx1=np.argsort(radiis)[:-alpha]
            X_new=X[idx1,:]
            weight2,center2=self.meepc(X_new)
            var2=np.square(X-center2)
            radiis=np.sqrt(np.matmul(weight2,var2.T))
            idx2=np.argsort(radiis)[:-alpha]
            #convergence
            if(np.all(idx1==idx2)):
                break


