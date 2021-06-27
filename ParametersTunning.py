import os
import time
import numpy as np

#Different Algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#Different CV methods
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from skopt.space import Real, Integer, Categorical
from skopt import BayesSearchCV
import warnings
warnings.filterwarnings('ignore')

from sklearn.externals import joblib


class ParametersTunning:

    FILE_PATH = os.path.dirname(os.path.abspath(__file__))
    TEST_DATA_PATH = os.path.abspath(os.path.join(FILE_PATH,'TEST_DATA'))
    CLASSIFIER_PATH = os.path.abspath(os.path.join(TEST_DATA_PATH,'CLASSIFIER'))
    MODEL_FOLDER_PATH = os.path.abspath(os.path.join(CLASSIFIER_PATH,'MODEL'))
    INFO_FOLDER_PATH = os.path.abspath(os.path.join(CLASSIFIER_PATH,'INFO')) 
    CM_REC_PATH = os.path.abspath(os.path.join(TEST_DATA_PATH,'CLUSTER_FOLDER/CLUSTER_REC_MODEL')) 

    #########################################################################################################
    ##########                                                                                     ##########
    ##########                         DEFINE THE HYPERPARAMETERS RANGE DICT                       ##########
    ##########                                                                                     ##########
    #########################################################################################################

    def parametersRangeDefine(self,recommender_name):

        if recommender_name is 'KNeighborsClassifier_uniform' :

            #Case 1
            hyperparameters_range_dictionary = {}
            hyperparameters_range_dictionary["n_neighbors"] = Integer(2, 30)
            hyperparameters_range_dictionary["algorithm"] = Categorical(['auto','ball_tree','kd_tree','brute'])
            # hyperparameters_range_dictionary["weight"] = Categorical(['uniform'])
            # hyperparameters_range_dictionary["n_jobs"] = Categorical([None,-1])
            # hyperparameters_range_dictionary["leaf_size"] = Categorical([30])
            # hyperparameters_range_dictionary["metric"] = Categorical(['minkowski','precomputed'])
            # hyperparameters_range_dictionary["metric_params"] = Categorical([None])
            # hyperparameters_range_dictionary["radius"] = Categorical([None])
            # hyperparameters_range_dictionary['p'] = Integer(1,6) 

            #class KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, 
            #                           metric='minkowski', metric_params=None, n_jobs=None, radius=None)

        if recommender_name is 'KNeighborsClassifier_distance' :
            #Case 2
            hyperparameters_range_dictionary = {}
            hyperparameters_range_dictionary["n_neighbors"] = Integer(2, 30)
            hyperparameters_range_dictionary["algorithm"] = Categorical(['auto','ball_tree','kd_tree','brute'])
            # hyperparameters_range_dictionary["weight"] = Categorical(['distance'])
            # hyperparameters_range_dictionary["n_jobs"] = Categorical([None,-1])
            # hyperparameters_range_dictionary["leaf_size"] = Categorical([30])
            hyperparameters_range_dictionary["p"] = Integer(1,6)
            # hyperparameters_range_dictionary["metric"] = Categorical(['minkowski','precomputed'])
            # hyperparameters_range_dictionary["metric_params"] = Categorical([None])
            # hyperparameters_range_dictionary["radius"] = Categorical([None])
            # hyperparameters_range_dictionary['p'] = Integer(1,6)  

        if recommender_name is 'SVC' :

            
            hyperparameters_range_dictionary = {}
            hyperparameters_range_dictionary["C"] = Real(low=1e-6, high=1e+6, prior='log-uniform')
            hyperparameters_range_dictionary["kernel"] = Categorical(['linear', 'poly', 'rbf'])
            hyperparameters_range_dictionary["degree"] = Integer(low=1,high=8)
            hyperparameters_range_dictionary["gamma"] = Real(low=1e-6, high=1e+1, prior='log-uniform')
            # hyperparameters_range_dictionary["coef0"] = Real(0,1,'log-uniform')
            # hyperparameters_range_dictionary["shrinking"] = Categorical([True,False])
            hyperparameters_range_dictionary["probability"] = Categorical([True]) 
            # hyperparameters_range_dictionary["tol"] = Real(1e-6,1e-3,'log-uniform')
            # hyperparameters_range_dictionary["cache_size"] = Integer(200,500)
            # hyperparameters_range_dictionary["class_weight"] = Categorical([None])
            # hyperparameters_range_dictionary["verbose"] = Categorical([True,False])
            # hyperparameters_range_dictionary["max_iter"] = Categorical([-1,30,50])
            # hyperparameters_range_dictionary["decision_function_shape"] = Categorical(['ovr','ovo'])

            # SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, 
            #     tol=1e-3, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', 
            #     break_ties=False, random_state=None)

        if recommender_name is 'DecisionTreeClassifier':
            hyperparameters_range_dictionary={}
            hyperparameters_range_dictionary['max_depth'] = Integer(1,5)
            hyperparameters_range_dictionary['criterion'] = Categorical(['gini','entropy'])
            hyperparameters_range_dictionary['max_features'] = Integer(1,40)
            hyperparameters_range_dictionary['max_leaf_nodes'] = Integer(2,20)
            hyperparameters_range_dictionary['min_samples_leaf'] = Integer(1,20)

        if recommender_name is 'LogisticRegression':
            hyperparameters_range_dictionary = {}
            hyperparameters_range_dictionary ['C']= Real(0.5,1)
            hyperparameters_range_dictionary['fit_intercept'] = Categorical([True,False])
            hyperparameters_range_dictionary['solver'] = Categorical(['newton-cg','lbfgs','liblinear','sag','saga'])
        
        if recommender_name is 'RandomForestClassifier' :
            hyperparameters_range_dictionary ={}
            hyperparameters_range_dictionary['n_estimators'] = Integer(10,100)
            hyperparameters_range_dictionary['criterion'] = Categorical(['gini','entropy'])
            hyperparameters_range_dictionary['max_depth'] = Integer(1,40)
            hyperparameters_range_dictionary['max_features'] = Integer(1,40)
            hyperparameters_range_dictionary['bootstrap'] = Categorical([True,False])

        return hyperparameters_range_dictionary

    #########################################################################################################
    ##########                                                                                     ##########
    ##########                        CROSS VALIDATION METHOD (CAN EXPAND LATER)                   ##########
    ##########                                                                                     ##########
    #########################################################################################################

    def bayes_search(self,estimator,X_train,y_train,parameter_space):
        bayes_search =BayesSearchCV(estimator,search_spaces=parameter_space,n_iter=2)
        bayes_search.fit(X_train,y_train) 
        return bayes_search

    #########################################################################################################
    ##########                                                                                     ##########
    ##########                 ACCORDING TO DIFFERENT CV METHODS TO SEARCH THE BEST MODEL          ##########
    ##########                                                                                     ##########
    #########################################################################################################

    def fit_Model(self,recommender_name,train_data,train_label,parameter_space):

        if recommender_name is 'KNeighborsClassifier_uniform' :
            estimator = KNeighborsClassifier(weights='uniform')
            
        if recommender_name is 'KNeighborsClassifier_distance':
            estimator = KNeighborsClassifier(weights='distance')

        if recommender_name is 'SVC':
            estimator = SVC()

        if recommender_name is 'DecisionTreeClassifier':
            estimator = DecisionTreeClassifier()

        if recommender_name is 'LogisticRegression':
            estimator = LogisticRegression()

        if recommender_name is 'RandomForestClassifier':
            estimator = RandomForestClassifier()

        start_time = time.time()
        model = self.bayes_search(estimator,train_data,train_label,parameter_space)
        train_time = time.time() - start_time

        print('Time Consuming for this fit is :{}\n'.format(train_time))

        #time

        return model,train_time

    #########################################################################################################
    ##########                                                                                     ##########
    ##########                   ACCORDING TO BEST ESTIMATOR TO GET THE BEST MODEL                 ##########
    ##########                                                                                     ##########
    #########################################################################################################

    def get_best_parameter(self,model):

        best_parameter = model.best_params_
        best_score = model.best_score_
        best_estimator =model.best_estimator_
        print("===> val. score: {}\n".format(best_score))
        print("===> best parameter: {}\n".format(best_parameter))
        print("===> Actual Estimator is: {}\n".format(best_estimator)) 
        return best_parameter,best_score,best_estimator

    def get_best_recommender(self,recommender_category,train_data,train_label,evaluation_data=None,evaluation_label=None,save_tag=True,save_file_name =None,file_path=None):
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Start Searching The best recommender<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')
        best_model_info = {}
        best_parameter = {}
        best_score = 0
        best_recommender = None
        best_timeconsuming = None

        if 'KNeighborsClassifier' in recommender_category :
            recommender_category.append('KNeighborsClassifier_uniform')
            recommender_category.append('KNeighborsClassifier_distance')
            recommender_category.remove('KNeighborsClassifier') 

        for recommender in recommender_category:
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
            print('Current Checking Recommender is >>>> [[ {} ]] \n'.format(recommender))
            hyperparameters_range_dictionary = self.parametersRangeDefine(recommender)
            model_tmp,timeconsuming = self.fit_Model(recommender,train_data,train_label,hyperparameters_range_dictionary)
            best_parameter_tmp,best_score_tmp,best_estimator_tmp = self.get_best_parameter(model_tmp)
            #ACC_EVAL_INFO
            acc_eval_info_tmp ={'AccScore':best_score_tmp,'Timeconsuming':timeconsuming,'EstimatorParam':best_estimator_tmp}
            best_model_info.update({recommender:acc_eval_info_tmp})

            evaluation_score = 0
            if evaluation_data is not None or evaluation_label is not None:
                evaluation_score = model_tmp.score(evaluation_data,evaluation_label)
                print('===> The Model evaluation score is: {}\n'.format(evaluation_score))

            if best_score_tmp > best_score :
                best_estimator = best_estimator_tmp
                best_parameter = best_parameter_tmp
                best_score = best_score_tmp
                best_recommender = recommender
                best_timeconsuming = timeconsuming
                best_evaluation_score = evaluation_score

            print('\n')
        best_model_info.update({'best_estimator':best_estimator,'best_tune_parameter':best_parameter,'best_score':best_score,'best_recommender':best_recommender,'best_timeconsuming':best_timeconsuming,'best_evaluation_score':best_evaluation_score})
        print('----------------------------------------------------------------------------------------------------------------\n')
        print("SEARCHING FINISH :\n The best recommder is {} \n its tune parameter is {} \n its score is {}\n its evaluation score is {} (0 means do not have the evaluation set)\n its model train time is {}\n The actually estimator is {}\n".format(best_recommender,best_parameter,best_score,best_evaluation_score,best_timeconsuming,best_estimator))
        print('----------------------------------------------------------------------------------------------------------------\n') 

        if save_tag == True:
            self.save_model(best_estimator,save_file_name,file_path)
            # self.save_info(best_model_info,save_file_name)
        return best_model_info

    #########################################################################################################
    ##########                                                                                     ##########
    ##########                 COMPUTE THE SCORE OF THE MODEL ON EVALUATION DATASET                ##########
    ##########                                                                                     ##########
    #########################################################################################################

    def evaluation_on_model(self,model,evaluation_data,evaluation_label):

        evaluation_score = model.score(evaluation_data, evaluation_label)

        print(">>> evaluation score:  {}\n".format(evaluation_score))

        return evaluation_score

    #########################################################################################################
    ##########                                                                                     ##########
    ##########           SAVE/LOAD MODEL AND INFO in DEFAULT PATH or file_path given by user       ##########
    ##########                                                                                     ##########
    #########################################################################################################

    def save_model (self,estimator,file_name,file_path=None):

        if file_path is None :
            if not os.path.exists(self.MODEL_FOLDER_PATH):
                os.makedirs(self.MODEL_FOLDER_PATH)

            file_path = self.MODEL_FOLDER_PATH + '/' + file_name + '.m'

        if file_path is 'cluster' :
            if not os.path.exists(self.CM_REC_PATH):
                os.makedirs(self.CM_REC_PATH)

            file_path = self.CM_REC_PATH + '/' + file_name + '.m' 

        joblib.dump(estimator,file_path)

        print('MODEL HAS BEEN SAVED IN ',file_path)

    def load_model(self,file_name,file_path=None):
        if file_path is None :
            file_path = self.MODEL_FOLDER_PATH + '/' + file_name + '.m'
        
        if file_path is 'cluster':
            file_path = self.CM_REC_PATH + '/' + file_name + '.m'

        estimator = joblib.load(file_path)
        return estimator

    def save_info(self,datainfo_dict,file_name,file_path=None):

        if file_path is None :
            if not os.path.exists(self.INFO_FOLDER_PATH):
                os.makedirs(self.INFO_FOLDER_PATH)
            file_path = self.INFO_FOLDER_PATH + '/' + file_name + '.npy' 
        
        if file_path is 'cluster' :
            if not os.path.exists(self.CM_REC_PATH):
                os.makedirs(self.CM_REC_PATH)

            file_path = self.CM_REC_PATH + '/' + file_name + '.npy'  

        np.save(file_path,datainfo_dict)
        print('INFO HAS BEEN SAVED IN ',file_path)

    def load_info(self,file_name,file_path=None):
        if file_path is None :
            file_path = self.INFO_FOLDER_PATH + '/' + file_name + '.npy'
        
        if file_path is 'cluster':
            file_path = self.CM_REC_PATH + '/' + file_name + '.npy'

        data_dict = np.load(file_path,allow_pickle=True).item()
        return data_dict

    #########################################################################################################
    ##########                                                                                     ##########
    ##########                 USE PREDICT() & PREDICT_PROBA() TO GET THE RESULT ON DATA TEST      ##########
    ##########                                                                                     ##########
    #########################################################################################################

    def get_prediction(self,model,data_input):
        predict_results = model.predict(data_input)
        return predict_results

    def get_score(self,model,data_input):
        probability = model.predict_proba(data_input)
        score = probability[:,1]
        return score.tolist()
    
    #########################################################################################################
    ##########                                                                                     ##########
    ##########                            USED TO TEST DIFFERENT ALGORITHM                         ##########
    ##########                                                                                     ##########
    #########################################################################################################
        
    def check(self,data):
        svc = SVC(probability=True)
        svc.fit(data['X_train'],data['y_train'])
        a = svc.predict_proba(data['X_test'])[:,1]

        print(a)

        knn = KNeighborsClassifier()
        knn.fit(data['X_train'],data['y_train'])
        b = knn.predict_proba(data['X_test'])[:,1]

        print(b)

        # det = DecisionTreeClassifier(min_samples_leaf=20)
        det = DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
                       max_depth=5, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=None, splitter='best')
        det.fit(data['X_train'],data['y_train'])
        c = det.predict_proba(data['X_test'])[:,1]

        print(c)

        log =LogisticRegression()
        log.fit(data['X_train'],data['y_train'])
        d = log.predict_proba(data['X_test'])[:,1]

        # print(d)

        ran = RandomForestClassifier()
        ran.fit(data['X_train'],data['y_train'])
        e = ran.predict_proba(data['X_test'])[:,1]

        print(e)



