import numpy as np
from scipy.linalg import svd
class RobustPCA:
  def __init__(self) -> None:
      pass

  def robust_pca(self,X, r, alpha=0.05, max_iter=100, tol=1e-4):
      n, d = X.shape
      alpha=int(alpha*n)
      U, Sigma, VT = svd(X)
      X_hat = np.dot(U[:, :r], np.dot( np.diag(Sigma[:r]) , VT[:r, :]))

      for i in range(max_iter):
          E = X - X_hat
          err = np.linalg.norm(E, axis=1) #reconstruction error

          # Set aside alpha fraction
          idx = np.argsort(err)[-alpha:]
          X_set = X[idx, :]
          E_set = E[idx, :]

          # indices of set aside pts.
          idx = np.setdiff1d(np.arange(n), idx)
          U_new, Sigma_new, VT_new = svd(X[idx, :])
          X_hat_new = np.dot(U_new[:, :r] , np.dot(np.diag(Sigma_new[:r]) , VT_new[:r, :]))

          # Compute SVD of set aside points
          U_set, Sigma_set, VT_set = svd(X_set)

          # Updation
          X_hat = X_hat_new + np.dot(E_set,VT_set.T)

          # Check convergence
          err_old = err
          err = np.linalg.norm(X - X_hat, axis=1)
          rel_err = np.linalg.norm(err - err_old) / np.linalg.norm(err_old)
          if rel_err < tol:
              break

      return X_hat, E
