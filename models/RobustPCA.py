import numpy as np
from scipy.linalg import svd
from copy import deepcopy
class RobustPCA:
  def __init__(self) -> None:
      pass

  def fit(self,X, r, alpha,Labels=None, max_iter=1000, tol=1e-4):
      n, d = X.shape
      alpha=int(alpha*n)
      U, Sigma, VT = svd(X)
      X_hat = U[:, :r]@ np.diag(Sigma[:r]) @ VT[:r, :]
      E = X - X_hat
      err = np.linalg.norm(E, axis=1) #reconstruction error
      V_old=VT
      if alpha == 0:
         return V_old
      for kk in range(max_iter):

        # Set aside alpha fraction
        inactive_idx = np.setdiff1d(np.arange(n), np.argsort(err)[:-alpha])
        X_inactive = X[inactive_idx, :]

        # indices of active pts.
        active_idx = np.setdiff1d(np.arange(n), inactive_idx)
        X_active = X[active_idx, :]

        # check for what % of attack points is considered inactive in this iteration

        if Labels is not None:


          attack_idx=np.where(Labels>0)[0]

          common_elements = np.isin( attack_idx , inactive_idx )
          # print("attack idx",attack_idx)
          # print("active idx",active_idx)
          percentage = np.count_nonzero(common_elements) / len(attack_idx) * 100

          print("Percentage of attack points considered inactive in {}th: (PCA) iteration is {:.2f} %".format(kk+1,percentage))

        U_new, Sigma_new, VT_new = svd(X_active)
        X_hat_active = np.dot(U_new[:, :r] , np.dot(np.diag(Sigma_new[:r]) , VT_new[:r, :]))


        # error computation for active points
        E_active = X_hat_active-X_active
        # reconstruction error of inactive points
        U_inactive=(np.linalg.pinv(np.diag(Sigma_new[:r])) @ VT_new[:r, :] @ X_inactive.T).T
        E_inactive = U_inactive @ np.diag(Sigma_new[:r]) @ VT_new[:r, :] - X_inactive
        # Check convergence
        if(np.all(np.all(V_old-VT_new,axis=1)<=tol)):
            break

        # Updation
        err = np.linalg.norm(np.concatenate((E_active,E_inactive),axis=0),axis=1)
        X = np.concatenate((X_active,X_inactive),axis=0)
        V_old=deepcopy(VT_new)
      return V_old
