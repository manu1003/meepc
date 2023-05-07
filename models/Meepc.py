import numpy as np
from copy import deepcopy
class MEEPC:
    def __init__(self) -> None:
        self.X = None
        self.rows = None
        self.cols = None
        # self.centroid = None
        # self.Z = None
        # self.alpha = None

    def calc_centroid(self):
        col_max = np.amax(self.X,axis=0)
        col_min = np.amin(self.X,axis=0)
        centroid = (col_max + col_min)/2
        return centroid
    
    def calc_Z(self,centroid):
        return (self.X-centroid)**2
    
    def calc_alpha(self):
        return (1.0 / self.rows) * np.ones(self.rows)

    def calc_h(self,alpha,Z):
        return np.matmul(alpha,Z)

    def isclose(self, a, b, rel_tol, abs_tol=0.0):
        return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    def calc_s(self,h,alpha_old,Z,u,i):
        ans=0
        for j in range(self.cols):
                    s_i_j = h[j] - alpha_old[i]*Z[i][j]
                    ans += (1/(u + s_i_j/Z[i][j]))
        return ans

    def binary(self,h,alpha_old,Z,i,low,high):
        mid=(low+high)/2
        temp_ans = self.calc_s(h,alpha_old,Z,mid,i)
        while(not self.isclose(temp_ans,1,1e-05)):
            # print(abs(temp_ans-1))
            if(temp_ans>1):
                low=mid
            elif(temp_ans<1):
                high=mid
            else:
                return mid
            mid = (low+high)/2
            temp_ans = self.calc_s(h,alpha_old,Z,mid,i)
        return mid

    def helper(self,h,alpha_old,Z,u,i):
        u_old=u # this 0.00001 is step which i am decreasing the u for binary search..
        
        while(self.calc_s(h,alpha_old,Z,u,i)>1):
            u_old = u
            u*=2        
        if u_old == u:
            u = self.binary(h,alpha_old,Z,i,0,1)
        else:
            u = self.binary(h,alpha_old,Z,i,u_old,u)    
        return u

    def fit(self,X,tol=1e-05):
        # print(X.shape)
        self.X = X
        self.rows,self.cols = X.shape
        centroid = self.calc_centroid()
        Z = self.calc_Z(centroid)
        alpha = self.calc_alpha()
        h = self.calc_h(alpha,Z)
        i = 0
        updates = 0
        converged = False
        alpha_old = np.zeros(self.rows)
        old = 0

        while not converged:
            for i in np.random.permutation(self.rows):
                alpha_old[i] = alpha[i]
                f_x = 0
                u=0
                f_x = self.calc_s(h,alpha_old,Z,u,i)
                
                if f_x == 1:
                    alpha[i] = u
                
                if f_x < 1:

                    alpha[i] = 0

                if f_x > 1:
                    alpha[i] = self.helper(h,alpha_old,Z,1,i)
                        
                h = h + (alpha[i] - alpha_old[i])*Z[i]
                # i += 1
                # if i>=self.rows-1 :
                #     i = 0
            updates += 1
            if old == 0:
                stored_alpha_old = deepcopy(alpha_old)
                old = 1
            if updates == 4:
                if np.all(np.abs(alpha - stored_alpha_old)) < tol :  #.00001
                    converged = True
                else:
                    updates = 0
                    stored_alpha_old = deepcopy(alpha)
        return 1/h,centroid
            
        # while i < self.rows:
        #     if converged:   
        #         # return weights
        #         return 1/h,centroid
        #         # return h,alpha
        #     alpha_old[i] = alpha[i]
        #     f_x = 0
        #     u=0
        #     f_x = self.calc_s(h,alpha_old,Z,u,i)
            
        #     if f_x == 1:
        #         alpha[i] = u
            
        #     if f_x < 1:

        #         alpha[i] = 0

        #     if f_x > 1:
        #         alpha[i] = self.helper(h,alpha_old,Z,1,i)
                    
        #     h = h + (alpha[i] - alpha_old[i])*Z[i]
        #     i += 1
        #     if i>=self.rows-1 :
        #         i = 0
        #         updates += 1
        #         if old == 0:
        #             stored_alpha_old = deepcopy(alpha_old)
        #             old = 1
        #     if updates == 4:
        #         if np.all(alpha - stored_alpha_old) < tol :  #.00001
        #             converged = True
        #         else:
        #             updates = 0
        #             stored_alpha_old = deepcopy(alpha)