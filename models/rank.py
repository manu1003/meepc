import numpy as np
import matplotlib.pyplot as plt
class Rank:
    def __init__(self) -> None:
        pass

    def fit(self,X,threshold=10):
        # Compute the eigenvalues and eigenvectors of the matrix
        eigenvalues, eigenvectors = np.linalg.eig(np.matmul(X, X.T))
        
        # Sort the eigenvalues in descending order
        sorted_eigenvalues = np.sort(eigenvalues)[::-1]

        # Plot the sorted eigenvalues
        plt.plot(np.log(sorted_eigenvalues))
        plt.xlabel('Eigenvalue Index')
        plt.ylabel('Eigenvalue')
        plt.show()

        # Look for the knee in the plot
        knee_index = np.argmin(np.abs(np.diff(sorted_eigenvalues)))

        # Compute the value of 'r'
        r = np.sum(sorted_eigenvalues[:knee_index + 1] >= threshold)

        return r

