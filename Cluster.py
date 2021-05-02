import numpy as np
import pandas as pd
import os
import shutil
import joblib

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

class CLUSTER:
    FILE_PATH = os.path.dirname(os.path.abspath(__file__))
    TEST_DATA_PATH = os.path.abspath(os.path.join(FILE_PATH,'TEST_DATA'))
    CLUSTER_FOLDER_PATH =os.path.abspath(os.path.join(TEST_DATA_PATH,'CLUSTER_FOLDER'))
    CLASSIFIER_FOLDER_PATH =os.path.abspath(os.path.join(TEST_DATA_PATH,'CLASSIFIER'))
    CM_FOLDER_PATH = os.path.abspath(os.path.join( CLUSTER_FOLDER_PATH,'CLUSTER_MODEL_INFO'))
    CM_REC_PATH =os.path.abspath(os.path.join( CLUSTER_FOLDER_PATH,'CLUSTER_REC_MODEL')) 
    MODEL_FOLDER_PATH =os.path.abspath(os.path.join( CLASSIFIER_FOLDER_PATH,'MODEL'))
    INFO_FOLDER_PATH =os.path.abspath(os.path.join( CLASSIFIER_FOLDER_PATH,'INFO'))

    #########################################################################################################
    ##########                                                                                     ##########
    ##########                            CLUSTER PARAMETERS TUNNING                               ##########
    ##########                                                                                     ##########
    #########################################################################################################

    def parameters_range_define(self,algorithm,tunning_method=None):
        if algorithm is 'KMeans' and tunning_method is 'elbow':
            cluster_parameters = {}
            cluster_parameters.update({'min_clusterNum':2})
            cluster_parameters.update({'max_clusterNum':50})

        if algorithm is 'KMeans' and tunning_method is None: #default==>'sihouette'
            cluster_parameters = {}
            cluster_parameters.update({'min_clusterNum':8})
            cluster_parameters.update({'max_clusterNum':15})

        if algorithm is 'DBSCAN':
            cluster_parameters = {}
            cluster_parameters.update({'min_eps':9})
            cluster_parameters.update({'max_eps':20})
            cluster_parameters.update({'step_eps':0.05})
            cluster_parameters.update({'min_Msample':5})
            cluster_parameters.update({'max_Msample':10})

        return cluster_parameters

    def cluster_parameters_tunning(self,data,cluster_parameters,algorithm,tunning_method=None):
        if algorithm is 'KMeans' and tunning_method is 'elbow':
            min_n = cluster_parameters['min_clusterNum']
            max_n = cluster_parameters['max_clusterNum']
            self.elbow_method(data,min_n,max_n)
            res = {}

        if algorithm is 'KMeans' and tunning_method is None:
            # default tunning_method is 'sihouette'
            min_n = cluster_parameters['min_clusterNum']
            max_n = cluster_parameters['max_clusterNum'] 
            bestK = self.silhouette_Coeficient_method(data, min_n, max_n)
            res={'algorithm':'KMeans','bestK':bestK}
            
        if algorithm is 'DBSCAN':
            min_eps = cluster_parameters['min_eps']
            max_eps = cluster_parameters['max_eps']
            step_eps = cluster_parameters['step_eps']
            min_Msample = cluster_parameters['min_Msample']
            max_Msample = cluster_parameters['max_Msample']
            eps,min_samples = self.parameterTunning_for_DBSCAN(X=data,min_eps=min_eps,max_eps=max_eps,step_eps=step_eps,min_Msamples=min_Msample,max_Msamples=max_Msample)
            res={'algorithm':'DBSCAN','eps':eps,'min_samples':min_samples}

        return res

    def elbow_method(self,df,min_n,max_n=50):

        if isinstance(df, pd.core.frame.DataFrame):
            df_x = preprocessing.scale(df.values)
            df =pd.DataFrame(data=df_x)

        #SSE(SUM of Square  Errors)
        inertia1 = []
        # Elbow Method
        for n in range(min_n, max_n):
            km = (KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=300, tol=0.0001, 
                        random_state=111, algorithm='elkan')) 

            km.fit(df)
            inertia1.append(km.inertia_)

        plt.figure(1, figsize=(15, 6))
        plt.plot(np.arange(min_n, max_n), inertia1, 'o')
        plt.plot(np.arange(min_n, max_n), inertia1, '-', alpha=0.7)
        plt.title('Elbow diagram', fontsize=12)
        plt.xlabel('Num of K'), plt.ylabel('SSE')
        plt.grid(linestyle='-.')
        plt.show()

        print("Done! The elbow diagram has been generated")

    def silhouette_Coeficient_method(self,X,min_n,max_n):
        #https://blog.csdn.net/weixin_26712065/article/details/108915871
        
        if isinstance(X, pd.core.frame.DataFrame):
            X = X.values
            X = preprocessing.scale(X)

        bestClusterNum = 0
        bestAvgSilhouetteScore = 0
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        for n_clusters in range(min_n,max_n):
            # Create a subplot with 1 row and 2 columns
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)
            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 
            ax1.set_xlim([-1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
            # Initialize the clusterer with n_clusters value and a random generator seed of 10 for reproducibility.
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(X)
            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed clusters
            silhouette_avg = silhouette_score(X, cluster_labels)
            print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
            if silhouette_avg > bestAvgSilhouetteScore :
                bestAvgSilhouetteScore = silhouette_avg
                bestClusterNum = n_clusters

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(X, cluster_labels)
            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
                ith_cluster_silhouette_values.sort()
                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i
                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)
                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")
            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')
            # Labeling the clusters
            centers = clusterer.cluster_centers_
            # Draw white circles at cluster centers
            ax2.scatter(centers[:, 0], centers[:, 1], marker='o', c="white", alpha=1, s=200, edgecolor='k')
            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50, edgecolor='k')

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")
            plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                        "with n_clusters = %d" % n_clusters),fontsize=14, fontweight='bold')
            plt.show()

        print('''===================================================================================
               \n After Comparing, the best cluster_number is {} 
               \n its avg silhouette score is {}
               \n
              '''.format(bestClusterNum,bestAvgSilhouetteScore))

        return bestClusterNum

    def parameterTunning_for_DBSCAN(self,X,min_eps=9,max_eps=10,step_eps=0.05,min_Msamples=5,max_Msamples=10):
        #https://blog.csdn.net/weixin_26712065/article/details/108915871
        
        if isinstance(X, pd.core.frame.DataFrame):
            df_x = preprocessing.scale(X.values)
            X =pd.DataFrame(data=df_x)

        best_score=0
        best_score_eps=0
        best_score_min_samples=0
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                #Build an empty list to save the results under different parameter combinations
        res = []
        for eps in np.arange(min_eps , max_eps , step_eps):
            #Iterating different values of  min_samples
            for min_samples in range(min_Msamples,max_Msamples):
                
                try:

                    dbscan = DBSCAN(eps = eps , min_samples = min_samples)
                    dbscan.fit(X)
                    labels = dbscan.labels_
                    #Count the number of clusters under each parameter combination (- 1 indicates abnormal point)
                    n_clusters = len([i for i in set(labels) if i != -1])
                    #Calculate Sihouette coeficient
                    score=silhouette_score(X,labels)
                    #Number of outliers and its ratio
                    outliners = np.sum(np.where(labels == -1 , 1 , 0))
                    ratio_outliners = outliners/len(labels)
                    raito = len(labels[labels[:] == -1]) / len(labels)
                    #Count the number of samples in each cluster
                    stats = str(pd.Series([i for i in dbscan.labels_ if i != -1]).value_counts().values)
                    res_tmp = {'eps':eps , 'min_samples':min_samples , 'n_clusters':n_clusters ,'score':score,'ratio_outliners':ratio_outliners, 'outliners':outliners , 'stats':stats}
                    res.append(res_tmp)

                    if score > best_score and n_clusters > 1:
                        best_score = score
                        best_score_eps =eps
                        best_score_min_samples = min_samples
                    
                    print('''\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    \n>>> Current parameters and its score:
                    \nEPS : {}    MIN_SAMPLES : {}
                    \nSCORE : {}

                    \nN_CLUSRERS : {} RATIO_OUTLINERS : {}
                    \nSTATS : {}

                    \n----------------------------------------------------------
                    \n>>>Curr BEST PARAMETERS:
                    \nBEST_EPS : {}
                    \nBEST_MIN_SAMPLES : {}
                    \nBEST_SCORE : {}
                    \n
                    '''.format(res_tmp['eps'],res_tmp['min_samples'],res_tmp['score'],res_tmp['n_clusters'],
                               res_tmp['ratio_outliners'],res_tmp['stats'],best_score_eps,best_score_min_samples,best_score))
        

                except Exception as ex:
                    # print('Wrong Parameter Set, Error MSG :',ex)
                    pass

        res = pd.DataFrame(data=res)
        # sns.relplot(x="eps",y="min_samples", size='score',data=res)
        # sns.relplot(x="eps",y="min_samples", size='raito_outliners',data=res)

        print(res.loc[res.score == best_score, :])

        print('''\n===============================================================
                \n>>> The BEST parameters for DBSCAN algorithm is:
                \n>>> eps : {}
                \n>>> min_samples: {}
                \n>>> its sihouette coeficient score is : {}
                
        '''.format(best_score_eps,best_score_min_samples,best_score))

        return best_score_eps,best_score_min_samples

    #########################################################################################################
    ##########                                                                                     ##########
    ##########                     CLUSTER MODEL FIT&SAVE AND GET ALL USER DISTRIBUTION            ##########
    ##########                                                                                     ##########
    #########################################################################################################

    def cluster_model_fit(self,data,algorithm,best_parameter,file_name=None,file_path=None,normalization='zero'):
        if normalization == 'zero':
            scaler = preprocessing.StandardScaler()
            scaler_file = 'Zero_cluserFitScaler'
        if normalization == 'minmax':
            scaler = preprocessing.MinMaxScaler()
            scaler_file == 'MinMax_clusterFitScaler'

        data = scaler.fit_transform(data)
        self.save_model(scaler, scaler_file)

        if algorithm is 'KMeans':
            bestK = best_parameter['bestK']
            km = KMeans(n_clusters=bestK, init='k-means++', n_init=10, max_iter=300,
                        tol=0.0001, random_state=111, algorithm='elkan').fit(data)
            data_pred = km.predict(data)
            clusterNum =bestK

            if file_name is None:
                file_name = 'KMeans_newest_cluster_model'
            self.save_model(km,file_name,file_path)

        if algorithm is 'DBSCAN':
            eps = best_parameter['eps']
            min_samples = best_parameter['min_samples']
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, algorithm='ball_tree', 
                            metric='euclidean').fit(data)
            data_pred = dbscan.labels_
            clusterNum = len([i for i in set(dbscan.labels_) if i != -1])

            if file_name is None:
                file_name = 'DBSCAN_newest_cluster_model'

            self.save_model(dbscan, file_name)
        
        result = {'algorithm':algorithm,'data_pred':data_pred,'clusterNum':clusterNum}
        return result
            
    def get_fitUserDistribution(self,df_userprofile,result):
        data_pred = result['data_pred']
        clusterNum = result['clusterNum']
        df_userprofile['clusterNo'] = data_pred

        cluster_res = {}
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
        for clusterNo in range(clusterNum):
            df_tmp = df_userprofile.loc[df_userprofile["clusterNo"]==clusterNo]
            users_list = list(df_tmp['User ID'].values)
            cluster_res.update({clusterNo:users_list})
            print('''=============================================================
                   \n>>> Cluster {} :
                   \n>>> Contains trained users :
                   \n>>> {}
                  '''.format(clusterNo,users_list))

        return cluster_res

    #########################################################################################################
    ##########                                                                                     ##########
    ##########                               PREDICT CLUSTER NO.                                   ##########
    ##########                                                                                     ##########
    #########################################################################################################

    def cluster_req_predict(self,req,algorithm,model_file=None,normalization='zero'):

        if normalization == 'zero':
            scaler_file = 'Zero_cluserFitScaler'
        if normalization == 'minmax':
            scaler_file == 'MinMax_clusterFitScaler' 
        scaler = self.load_model(scaler_file)
        req = scaler.transform(req)

        if algorithm is 'KMeans':
            if model_file is None:
                model_file = 'KMeans_newest_cluster_model'
                model = self.load_model(model_file)
                clusterNo = model.predict(req)
        
        if algorithm is 'DBSCAN':
            if model_file is None:
                model_file = 'DBSCAN_newest_cluster_model'
                dbscan = self.load_model(model_file)
                clusterNum = len([i for i in set(dbscan.labels_) if i != -1])

                knn = KNeighborsClassifier(n_neighbors=clusterNum)
                knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_])
                clusterNo = knn.predict(req)

        print('The ClusterNo for the user is :',clusterNo)

        return clusterNo

    #########################################################################################################
    ##########                                                                                     ##########
    ##########                                  SUPORT  FUNCTIONS                                  ##########
    ##########                                                                                     ##########
    #########################################################################################################

    def save_model (self,estimator,file_name,file_path=None):

        if file_path is None :
            if not os.path.exists(self.CM_FOLDER_PATH):
                os.makedirs(self.CM_FOLDER_PATH)

            file_path = self.CM_FOLDER_PATH + '/' + file_name + '.m'

        joblib.dump(estimator,file_path)

        print('CLUSTER MODEL HAS BEEN SAVED IN ',file_path)

    def load_model(self,file_name,file_path=None):
        if file_path is None :
            file_path = self.CM_FOLDER_PATH + '/' + file_name + '.m'

        estimator = joblib.load(file_path)
        return estimator

    def copy_model(self,source_file,target_file,source_path=None,target_path='model'):

        if source_path is None:
            source_path = self.CM_REC_PATH
        if target_path is 'model':
            target_path = self.MODEL_FOLDER_PATH
        if target_path is 'info' :
            target_path=self.INFO_FOLDER_PATH

        shutil.copy(source_path + '/' + source_file, target_path + '/' + target_file)
        
    def save_info(self,datainfo_dict,file_name,file_path='clusterM'):
        
        if file_path is 'clusterM' :
            if not os.path.exists(self.CM_FOLDER_PATH):
                os.makedirs(self.CM_FOLDER_PATH)

            file_path = self.CM_FOLDER_PATH + '/' + file_name + '.npy'  

        np.save(file_path,datainfo_dict)
        print('INFO HAS BEEN SAVED IN ',file_path)

    def load_info(self,file_name,file_path='clusterM'):        
        if file_path is 'clusterM':
            file_path = self.CM_FOLDER_PATH + '/' + file_name + '.npy'

        data_dict = np.load(file_path,allow_pickle=True).item()
        return data_dict


# a = CLUSTER()
# print(a.INFO_FOLDER_PATH)
