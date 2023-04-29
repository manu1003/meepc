import numpy as np
from scipy.linalg import svd
from copy import deepcopy
class RobustPCA:
  def __init__(self) -> None:
      pass

  def fit(self,X, r, alpha=0.05, max_iter=1000, tol=1e-4):
      n, d = X.shape
      alpha=int(alpha*n)
      U, Sigma, VT = svd(X)
      X_hat = np.dot(U[:, :r], np.dot( np.diag(Sigma[:r]) , VT[:r, :]))
      X_old=X
      for i in range(max_iter):
          E = X_old - X_hat  #doubt
          err = np.linalg.norm(E, axis=1) #reconstruction error

          # Set aside alpha fraction
          idx = np.argsort(err)[-alpha:]
          X_set = X[idx, :]
          E_set = E[idx, :]

          # indices of active pts.
          idx = np.setdiff1d(np.arange(n), idx)
          U_new, Sigma_new, VT_new = svd(X[idx, :])
          X_hat_new = np.dot(U_new[:, :r] , np.dot(np.diag(Sigma_new[:r]) , VT_new[:r, :]))

          # Compute SVD of set aside points
          U_set, Sigma_set, VT_set = svd(X_set)
          X_old=deepcopy(X_hat)
          err_old = deepcopy(err)

          # Updation
          X_hat = X_hat_new + np.dot(U_set,np.dot(Sigma_set,E_set))  #doubt


          # Check convergence
          err = np.linalg.norm(X_old - X_hat, axis=1)
          if(np.all(np.all(err-err_old,axis=1))<tol):
              break

      return X_hat,E,VT_new
