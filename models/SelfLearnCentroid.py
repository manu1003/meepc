import numpy as np
class SelfLearnCentroid:

    def __init__(self) -> None:
        pass

    def fit(P, tol = 0.001):

        d, N = P.shape
        
        Q = np.zeros((d+1, N))
        Q[0:d, :] = P
        Q[d, :] = np.ones((1, N))

        count = 1
        err = 1
        u = (1.0/N)*np.ones(N)
        u = u.transpose()

        while err > tol:

            X = np.dot(np.dot(Q, np.diag(u)), np.transpose(Q))

            try:
                invX = np.linalg.inv(X)
                M = np.diag(np.dot(np.dot(np.transpose(Q), invX), Q))
                maxM = np.amax(M)
                j = np.argmax(M)
                step_size = (maxM - d - 1)/((d+1)*(maxM - 1))
                new_u = (1 - step_size)*u
                new_u[j] = new_u[j] + step_size
                count += 1
                err = np.linalg.norm(new_u - u)
                u = new_u

            except np.linalg.LinAlgError as e:
                if 'Singular matrix' in str(e):
                    # If X is singular, find its null space and set u to be orthogonal
                    # to all of its basis vectors
                    _, s, vh = np.linalg.svd(X)
                    null_mask = (s <= np.finfo(float).eps)
                    null_space = vh[null_mask].T
                    if null_space.size > 0:
                        u = u - np.dot(u, null_space).dot(null_space.T)
                        u = u / np.linalg.norm(u)
                else:
                    print('Error:', e)
                    return None
        MatU = np.diag(u)

        u.transpose()
        centroid = np.dot(P, u)
        
        return centroid
