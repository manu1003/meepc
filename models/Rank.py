import numpy as np
import matplotlib.pyplot as plt
class Rank:
    def __init__(self) -> None:
        pass

    def fit(self,X,threshold=10):
        # Compute the eigenvalues and eigenvectors of the matrix
        eigenvalues, eigenvectors = np.linalg.eig(np.matmul(X, X.T))
        # print("eigen values")
        # print(eigenvalues)
        # Sort the eigenvalues in descending order
        sorted_eigenvalues = np.sort(eigenvalues)[::-1]

        # Plot the sorted eigenvalues
        # plt.plot(np.log(sorted_eigenvalues))
        # plt.xlabel('Eigenvalue Index')
        # plt.ylabel('Eigenvalue')
        # # plt.xlim(0,2)
        # plt.show()

        # Look for the knee in the plot
        # knee_index = np.argmin(np.abs(np.diff(sorted_eigenvalues)))

        # # Compute the value of 'r'
        # r = np.sum(sorted_eigenvalues[:knee_index + 1] >= threshold)


        if len(sorted_eigenvalues)>4 and np.any(sorted_eigenvalues)>0:
            thresh=0.1*sorted_eigenvalues[4]
            indices=np.where(sorted_eigenvalues>thresh)[0]
            r=len((list(indices)))
        else:
            # r=len(sorted_eigenvalues)-1
              r=len(sorted_eigenvalues)
        return r

