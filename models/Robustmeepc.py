import numpy as np
from models import MEEPC

class Robustmeepc:

    def __init__(self) -> None:
        self.meepc=MEEPC()

        pass

    def fit(self,X,alpha_factor,Labels=None,cluster_index=None,beta_factor=.1):
        n,d = X.shape
        alpha = int(alpha_factor*n)
        beta = round(beta_factor*alpha)
        total_common = 0
        total_attack_meepc=[]
        print("robust meepc iterations")
        if beta != 0:
            max_iter = round(1/beta_factor)

            for i in range(max_iter):
                weight,center = self.meepc.fit(X)
                #calculating radiis for all points
                var1 = np.square(X-center)
                radii = np.sqrt(np.matmul(weight,var1.T))
                idx = np.setdiff1d(np.arange(len(X)),np.argsort(radii)[-beta:])
                X = X[idx]

                if Labels is not None :
                    attack_idx=np.where(Labels>0)[0]
                    if len(attack_idx) != 0:
                        alpha_idx = np.argsort(radii)[-beta:]
                        common_elements = np.intersect1d( attack_idx , alpha_idx )
                        total_common+=len(common_elements)
                        print("recall in this iteration",total_common/len(attack_idx))


                    Labels=Labels[idx]
                    total_attack_meepc.append(cluster_index[alpha_idx])
                    cluster_index=cluster_index[idx]
        weight,center = self.meepc.fit(X)
        if Labels is not None:
            print(len(total_attack_meepc))
            print("------[Meepc] attack points found is {} ".format(total_common))

        return weight,center,total_common,total_attack_meepc



