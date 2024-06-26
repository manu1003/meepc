import numpy as np
from models import Hankel,Rank,Cluster,Robustmeepc,RobustPCA,Robustcluster
import warnings
warnings.simplefilter('ignore')
from sklearn.cluster import KMeans
from gekko import GEKKO
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class RobustPipeline:
    def __init__(self) -> None:
        self.hankel = Hankel()
        self.rank = Rank()
        self.cluster = Cluster()
        self.robustcluster= Robustcluster()
        self.meepc = Robustmeepc()
        self.pca=RobustPCA()
        self.lag = None
        self.stride = None
        self.only_corr = False
        self.optimal_k = None
        self.cluster_centers = None
        self.labels=None
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
        self.y_train= None
        self.cluster_idx=None
        self.pca_idx= []
        self.counts_meepc=0
        self.attack_idx_k_wise = []
        self.total_pca_attacks=0
        self.total_meepc_attacks=[]


    def getCenter(self,X):
        A=((X.max(axis=0)-X.min(axis=0))/2)+0.00001
        C=(X.max(axis=0)+X.min(axis=0))/2
        return A,C


    def getW(self,X):
        N=X.shape[0]
        d=X.shape[1]
        # print(N,d)
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
        # return min( np.max( radii_n ), radiis[indices[idx]])
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

    def calc_normal_variables(self,X,alpha):
        radii_normal = []
        attack_points=[]
        # Store the indices of data points assigned to each cluster

        cluster_indices = [[] for _ in range(len(self.labels))]
        for i, assignment in enumerate(self.labels):
            cluster_indices[assignment].append(i)


        for i in np.unique(self.labels):
            cluster_ = X[np.where(self.labels == i)[0]]
            r = self.rank.fit(cluster_)

            self.clusterR.append(r)
        #PCA
            if self.y_train is not None:
                VT,pca_count,pca_alpha_idx= self.pca.fit(cluster_,r,alpha,Labels=self.y_train[np.where(self.labels == i)[0]],cluster_index=np.asarray(cluster_indices[i]))
                # print("length of pca active indx are ",len(pca_alpha_idx))
                self.total_pca_attacks += pca_count
                self.pca_idx.append(pca_alpha_idx)
                # print("total attack found yet",self.total_pca_attacks)
            else:
                VT,temp1 = self.pca.fit(cluster_,r,alpha)

            V = VT.T
            self.clusterV.append(V)
            cluster_ = np.matmul(cluster_,V[:,:r])
        #MEEPC
            if self.use_gekko:
                weight,center = self.getW(cluster_)
            else:
                if self.y_train is not None:
                    weight,center,counts,meepc_attacks = self.meepc.fit(cluster_,alpha,Labels=self.y_train[np.where(self.labels == i)[0]],cluster_index=np.asarray(cluster_indices[i]))
                    self.counts_meepc += counts
                    self.total_meepc_attacks.append(meepc_attacks)

                else:
                    weight,center,temp2,temp5 = self.meepc.fit(cluster_,alpha)

            self.weights.append(weight)
            self.centers.append(center)
            var1=np.square(cluster_-center)
            var2=np.matmul(weight,var1.T)
            radii_normal.append(np.sqrt(var2))



        # Map the indexes of attack points back to the original dataset
        mapped_attack_indices = []

        # for cluster_index, attack_indices in enumerate(attack_points):
        #     print()
        #     original_indices = [cluster_indices[cluster_index][attack_index] for attack_index in attack_indices]
        #     mapped_attack_indices.append(original_indices)
        # self.pca_idx = mapped_attack_indices

        return radii_normal


    def get_data(self,X,corr=None,y_truth=None):

        if y_truth is not None:
            labels = self.hankel.fit(np.array(y_truth),self.lag,self.stride)
            self.y_train = np.any(labels>0,axis=0).astype(int)

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
            corr_attack=None,only_corr=False,use_gekko=False,alpha=None,y_truth=None):
        self.lag = lag
        self.stride = stride
        self.only_corr = only_corr
        self.use_gekko = use_gekko


        # train on normal data and get all required variables on it
        X = self.get_data(train_normal,corr_normal,y_truth=y_truth)

        optimal_k = min(optimal_k,len(np.unique(X)))
        if y_truth is not None:
            attack_idx=np.where(self.y_train>0)[0]
            # print("Total attack data points is {} out of {}".format(len(attack_idx),len(self.y_train)))
            self.cluster_centers,self.labels,self.cluster_idx = self.robustcluster.fit(X,optimal_k,alpha,Labels = self.y_train)


        else:
            self.cluster_centers,self.labels,_ = self.robustcluster.fit(X,optimal_k,alpha)

        # print("Clustering Done")
        self.optimal_k = len(np.unique(self.labels))



        self.radii_normal = self.calc_normal_variables(X,alpha)

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










