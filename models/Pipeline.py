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
        self.accuracy = []
        self.precision = []
        self.recall = []
        self.fscore = []
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
    
    def calc_threshold(self,optimal_k,radii_normal,radii_attack):
        threshold_clusters = [0]*optimal_k
        for i in range(optimal_k):
            threshold_clusters[i] = self.tune_threshold(radii_normal[i],radii_attack[i])
        return threshold_clusters
    
    def calc_distances(self,X,optimal_k,weights,centers,clusters_V,clusters_R):
        distances = []
        for i in range(optimal_k):
            cluster_ = np.matmul(X,clusters_V[i][:,:clusters_R[i]])
            var1=np.square(cluster_- centers[i])
            var2=np.matmul(weights[i],var1.T)
            distances.append(np.sqrt(var2))    
        return distances
    
    
    def calc_normal_variables(self,X,kmeans,optimal_k):
        radii_normal = []
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
            radii_normal.append(np.sqrt(var2))
        return radii_normal,weights,centers,clusters_V,clusters_R
    
    def get_data(self,df,sens,lag,stride):
        X = df.iloc[:,sens].values
        X = self.hankel.fit(X,lag,stride)
        X = X.T
        return X
    
    def fit(self,train_normal,train_attack,test,y_actual,lag,stride,optimal_k = None):

        for sens in range(len(train_normal.columns)):

            # train on normal data and get all required variables on it
            X = self.get_data(train_normal,sens,lag,stride)
            if not optimal_k:
                kmeans,optimal_k = self.cluster.fit(X)
                kmeans.fit(X)
            else:
                kmeans = KMeans(n_clusters=optimal_k,init='k-means++')
                kmeans.fit(X)
            radii_normal,weights,centers,clusters_V,clusters_R = self.calc_normal_variables(X,kmeans,optimal_k)

            # use attack data in train data to tune the threshold
            X_att = self.get_data(train_attack,sens,lag,stride)
            radii_attack = self.calc_distances(X_att,optimal_k,weights,centers,clusters_V,clusters_R)

            # calculate the thresholds
            threshold_clusters = self.calc_threshold(optimal_k,radii_normal,radii_attack)


            X_test = self.get_data(test,sens,lag,stride)
            radii_test = self.calc_distances(X_test,optimal_k,weights,centers,clusters_V,clusters_R)
            radii_test = np.transpose(np.vstack(radii_test))

            # calc diff to find anomalies if any
            check_anomaly = threshold_clusters-radii_test

            # if outside all clusters then anomaly
            y_predicted = np.all(check_anomaly<0,axis=1).astype(int)

            # used to calc auc-roc score
            y_score = np.min((radii_test/threshold_clusters),axis=1)

            self.accuracy.append(accuracy_score(y_actual,y_predicted))
            self.precision.append(precision_score(y_actual,y_predicted))
            self.recall(recall_score(y_actual,y_predicted))
            self.fscore(f1_score(y_actual,y_predicted))
            


            


            

        

