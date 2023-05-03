import numpy as np
from models import Hankel,Rank,Cluster,MEEPC
import warnings
warnings.simplefilter('ignore')
from sklearn.cluster import KMeans
from gekko import GEKKO
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class Pipeline:
    def __init__(self) -> None:
        self.hankel = Hankel()
        self.rank = Rank()
        self.cluster = Cluster()
        self.meepc = MEEPC()
        self.lag = None
        self.stride = None
        self.only_corr = False
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

    def getCenter(self,X):
        A=((X.max(axis=0)-X.min(axis=0))/2)+0.00001
        C=(X.max(axis=0)+X.min(axis=0))/2
        return A,C


    def getW(self,X):
        N=X.shape[0]
        d=X.shape[1]
        print(N,d)
        m = GEKKO(remote=False)
        m.options.MAX_ITER=1000
        w = m.Array(m.Var,d,lb=0)
    #    X_nor_sig=np.copy(X)
        A,C = self.getCenter(X)
        for i in range(d):
            w[i].value=(2*A[i])**-0.5
        X_centred=(X-C)**2
        for i in range(N):
            m.Equation(np.dot(w,X_centred[i])<=1)
        prod=w[0]
        for i in range(d-1):
            prod=prod*w[i+1]
        m.Obj(prod)
        m.solve(disp=False)
        weight=np.zeros(d,dtype=float)
        for i in range(len(weight)):
            weight[i]=w[i].value[0]
        return weight,C

    def tune_threshold(self,radii_n,radii_a):
        label = [-1]*len(radii_n) + [1]*len(radii_a)
        label = np.array(label)
        radiis = np.array(list(radii_n)+list(radii_a))
        indices = np.argsort(radiis)
        label = label[indices]
        pos_temp = (1+label)//2
        neg_temp = (1-label)//2
        n_pos = len(radii_a)
        n_neg = len(radii_n)
        fn = np.cumsum(pos_temp)
        tp = n_pos - fn
        fp = n_neg - np.cumsum(neg_temp)
        fmeas = (2*tp )/ (2*tp + fp + fn)
        idx = np.argmax(fmeas)
        # return min( np.max( radii_n ), radiis[indices[idx]] )
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
        for i in range(self.optimal_k):
            cluster_ = X[np.where(kmeans.labels_ == i)[0]]
            r = self.rank.fit(cluster_)
            self.clusterR.append(r)
            U,Sigma,VT = np.linalg.svd(cluster_)
            V = VT.T
            self.clusterV.append(V)
            cluster_ = np.matmul(cluster_,V[:,:r])
            if self.use_gekko:
                weight,center = self.getW(cluster_)
            else:
                weight,center = self.meepc.fit(cluster_)
            self.weights.append(weight)
            self.centers.append(center)
            var1=np.square(cluster_-center)
            var2=np.matmul(weight,var1.T)
            radii_normal.append(np.sqrt(var2))
        return radii_normal


    def get_data(self,X,corr=None):
        if self.only_corr:
            if corr is not None:
                X = self.hankel.fit(X,self.lag,self.stride)
                mini=X.min(axis=0).reshape(1,-1)
                maxi=X.max(axis=0).reshape(1,-1)
                avg=X.mean(axis=0).reshape(1,-1)
                corr=np.concatenate((corr,mini),axis=0)
                corr=np.concatenate((corr,maxi),axis=0)
                corr=np.concatenate((corr,avg),axis=0)
                return corr.T
            else:
                print('No correlation matrix given to create hankel')
                return
        # X = df.iloc[:,sens].values
        X = self.hankel.fit(X,self.lag,self.stride)
        if (corr is not None):
            X=np.concatenate((X,corr),axis=0)
        X = X.T
        return X

    def fit(self,train_normal,train_attack,lag,stride,optimal_k = None,kscore_init='silhouette',tune=True,corr_normal=None,
            corr_attack=None,only_corr=False,use_gekko=False):
        self.lag = lag
        self.stride = stride
        self.only_corr = only_corr
        self.use_gekko = use_gekko
        # for sens in range(len(train_normal.columns)):

        # train on normal data and get all required variables on it
        X = self.get_data(train_normal,corr_normal)
        # ,sens)
        if not optimal_k:
            kmeans,optimal_k = self.cluster.fit(X,kscore_init)
            kmeans.fit(X)
        else:
            optimal_k = min(optimal_k,len(np.unique(X)))
            kmeans = KMeans(n_clusters=optimal_k,init='k-means++')
            kmeans.fit(X)
        self.optimal_k = optimal_k
        # print("optimalK",optimal_k)
        self.radii_normal = self.calc_normal_variables(X,kmeans)
        # ,weights,centers,clusters_V,clusters_R
        # use attack data in train data to tune the threshold
        if tune:
            X_att = self.get_data(train_attack,corr_attack)
            self.radii_attack = self.calc_distances(X_att)

            # calculate the thresholds
            self.threshold_clusters = np.asarray(self.calc_threshold())
        else:
            self.threshold_clusters = np.asarray([np.max(self.radii_normal[i]) for i in range(self.optimal_k)])


    def predict(self,X_test,corr_test=None):

        X_test = self.get_data(X_test,corr_test)
        radii_test = self.calc_distances(X_test)
        self.radii_test = np.transpose(np.vstack(radii_test))

        # used to calc auc-roc score
        self.y_score = np.min((self.radii_test/self.threshold_clusters),axis=1)

        # calc diff to find anomalies if any
        self.check_anomaly = self.threshold_clusters-self.radii_test

        # if outside all clusters then anomaly
        self.y_predicted = np.all(self.check_anomaly<0,axis=1).astype(int)

        return self.y_predicted










