import numpy as np
from models import MEEPC

class Robustmeepc:

    def __init__(self) -> None:
        self.meepc=MEEPC()
        pass

    def fit(self,X,alpha_factor,Labels=None,beta_factor=.1):
        n,d = X.shape
        alpha = int(alpha_factor*n)
        beta = round(beta_factor*alpha)
        if beta != 0:
            max_iter = round(1/beta_factor)
            for i in range(max_iter):
                weight,center = self.meepc.fit(X)
                #calculating radiis for all points
                var1 = np.square(X-center)
                radii = np.sqrt(np.matmul(weight,var1.T))
                idx = np.setdiff1d(np.arange(len(X)),np.argsort(radii)[-beta:])
                X = X[idx]
                if Labels is not None:

                    attack_idx=np.where(Labels>0)[0]

                    common_elements = np.isin( attack_idx , np.argsort(radii)[-beta:] )
                    print("common elements",common_elements)
                    if len(attack_idx) != 0:

                        percentage = np.count_nonzero(common_elements) / len(attack_idx) * 100

                        print("Percentage of attack points considered inactive in {}th: (MEEPC) iteration is {:.2f} %".format(i+1,percentage))

        weight,center = self.meepc.fit(X)
        return weight,center



