import numpy as np
from models import Hankel,Rank,Cluster,Meepc
import warnings
warnings.simplefilter('ignore')
from sklearn.cluster import KMeans
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class Pipeline:
    def __init__(self) -> None:
        self.hankel = Hankel()
        self.rank = Rank()
        self.cluster = Cluster()
        self.meepc = Meepc.MEEPC()
        self.lag = None
        self.stride = None
        self.optimal_k = None
        self.weights = []
        self.centers = []
        self.clusterV = []
        self.clusterR = []
        self.radii_normal = None
        self.radii_attack = None
        self.radii_test = None
        self.threshold_clusters = None
        self.check_anomaly = None
        self.y_predicted = None
        self.y_score = None
        # self.accuracy = []
        # self.precision = []
        # self.recall = []
        # self.fscore = []
        # pass

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
        idx = np.argmax(fmeas)
        return radiis[indices[idx]]
    
    def calc_threshold(self):
        threshold_clusters = [0]*self.optimal_k
        for i in range(self.optimal_k):
            threshold_clusters[i] = self.tune_threshold(self.radii_normal[i],self.radii_attack[i])
        return threshold_clusters
    
    def calc_distances(self,X):
        distances = []
        for i in range(self.optimal_k):
            cluster = np.matmul(X,self.clusterV[i][:,:self.clusterR[i]])
            var1=np.square(cluster - self.centers[i])
            var2=np.matmul(self.weights[i],var1.T)
            distances.append(np.sqrt(var2))    
        return distances    
    
    def calc_normal_variables(self,X,kmeans):
        radii_normal = []
        # weights=[]
        # centers=[]
        # clusters_R=[]
        # clusters_V=[]
        for i in range(self.optimal_k):
            cluster_ = X[np.where(kmeans.labels_ == i)[0]]
            r = self.rank.fit(cluster_)
            self.clusterR.append(r)
            U,Sigma,VT = np.linalg.svd(cluster_)
            V = VT.T
            self.clusterV.append(V)
            cluster_ = np.matmul(cluster_,V[:,:r])
            weight,center = self.meepc.fit(cluster_)
            self.weights.append(weight)
            self.centers.append(center)
            var1=np.square(cluster_-center)
            var2=np.matmul(weight,var1.T)
            radii_normal.append(np.sqrt(var2))
        return radii_normal
    # ,weights,centers,clusters_V,clusters_R
    
    def get_data(self,X):
        # X = df.iloc[:,sens].values
        X = self.hankel.fit(X,self.lag,self.stride)
        X = X.T
        return X
    
    def fit(self,train_normal,train_attack,lag,stride,optimal_k = None,tune=True):

        self.lag = lag
        self.stride = stride

        # for sens in range(len(train_normal.columns)):

        # train on normal data and get all required variables on it
        X = self.get_data(train_normal)
        # ,sens)
        if not optimal_k:
            kmeans,optimal_k = self.cluster.fit(X)
            kmeans.fit(X)
        else:
            kmeans = KMeans(n_clusters=optimal_k,init='k-means++')
            kmeans.fit(X)
        self.optimal_k = optimal_k
        self.radii_normal = self.calc_normal_variables(X,kmeans)
        # ,weights,centers,clusters_V,clusters_R
        # use attack data in train data to tune the threshold
        if tune:
            X_att = self.get_data(train_attack)
            self.radii_attack = self.calc_distances(X_att)

            # calculate the thresholds
            self.threshold_clusters = np.asarray(self.calc_threshold())
        else:
            self.threshold_clusters = np.max(np.array(self.radii_normal))

    
    def predict(self,X_test):

        X_test = self.get_data(X_test)
        radii_test = self.calc_distances(X_test)
        self.radii_test = np.transpose(np.vstack(radii_test))

        # used to calc auc-roc score
        self.y_score = np.min((self.radii_test/self.threshold_clusters),axis=1)

        # calc diff to find anomalies if any
        self.check_anomaly = self.threshold_clusters-self.radii_test

        # if outside all clusters then anomaly
        self.y_predicted = np.all(self.check_anomaly<0,axis=1).astype(int)

        return self.y_predicted
    

        

        # self.accuracy.append(accuracy_score(y_actual,y_predicted))
        # self.precision.append(precision_score(y_actual,y_predicted))
        # self.recall(recall_score(y_actual,y_predicted))
        # self.fscore(f1_score(y_actual,y_predicted))
        # return self
            


            


            

        

