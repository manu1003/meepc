import numpy as np
import pandas as pd
import Hankel,Rank,Cluster,Meepc
import warnings
warnings.simplefilter('ignore')
import math
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class Pipeline:
    def __init__(self) -> None:
        self.hankel = Hankel()
        self.rank = Rank()
        self.cluster = Cluster()
        self.meepc = Meepc.MEEPC()
        self.scaler = StandardScaler()
        pass

    def tune_threshold(self,radii_n,radii_a):
        label = [1]*len(radii_n) + [-1]*len(radii_a)
        label = np.array(label)
        radiis = np.array(list(radii_n)+list(radii_a))
        indices = np.argsort(radiis)
        label = label[indices]
        pos_temp = (1+label)//2
        neg_temp = (1-label)//2
        tp = np.cumsum(pos_temp)
        fp = np.cumsum(neg_temp)
        fn = tp[-1] - tp
        fmeas = 2*tp / (2*tp + fp + fn)
        # print(fmeas)
        idx = np.argmax(fmeas)
        # print(indices[idx])
        return radiis[indices[idx]]
    
    def calc_threshold(self):
        X_attack = df_attack.iloc[:,sens].values
        X_att = hankel.fit(X_attack,1,0.5)
        X_att=X_att.T 
        radiis_attack = []
        for i in range(optimal_k):
            cluster_ = np.matmul(X_att,clusters_V[i][:,:clusters_R[i]])
            var1=np.square(cluster_-centers[i])
            var2=np.matmul(weights[i],var1.T)
            radiis_attack.append(np.sqrt(var2))
        threshold_clusters = [0]*optimal_k
        for i in range(optimal_k):
            threshold_clusters[i] = self.tune_threshold(radiis_normal[i],radiis_attack[i])
    
    def calc_normal_radius(self,train_normal,lag,stride,optimal_k = None):

        for sens in range(len(train_normal.columns)):
            X = train_normal.iloc[:,sens].values
            X = self.hankel.fit(X,lag,stride)
            X = X.T 
            if not optimal_k:
                kmeans,optimal_k = self.cluster.fit(X)
                kmeans.fit(X)
            else:
                kmeans = KMeans(n_clusters=optimal_k,init='k-means++')
                kmeans.fit(X)
            
            radiis_normal = []
            weights=[]
            centers=[]
            clusters_R=[]
            clusters_V=[]
            for i in range(optimal_k):
                cluster_ = X[np.where(kmeans.labels_ == i)[0]]
                r = self.rank.fit(cluster_)
                clusters_R.append(r)
                if(optimal_k==1):
                    print("r: "+str(r))
                U,Sigma,VT = np.linalg.svd(cluster_)
                V = VT.T
                clusters_V.append(V)
                cluster_ = np.matmul(cluster_,V[:,:r])
                weight,center = self.meepc.fit(cluster_)
                weights.append(weight)
                centers.append(center)
                var1=np.square(cluster_-center)
                var2=np.matmul(weight,var1.T)
                radiis_normal.append(np.sqrt(var2))
    
    def fit(self,train_normal,train_attack,test,cluster_no = None):

        for sens in range(len(train_normal.columns)):

        

