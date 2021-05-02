import os
import time
import datetime
import random
import pandas as pd

from shutil import copyfile

from ParametersTunning import ParametersTunning
from Step2_TSP2OfferCategorizer import TSP2OfferCategorizer
from LEARNER import DATA_Population
from LEARNER import TEST_HistoricalDataGenerate
from Cluster import CLUSTER


class GET_UserProfile:

    his = TEST_HistoricalDataGenerate()
    
    def _getUserCurrProfile(self,username):
        profile_path = self.his.TEST_USER_PROFILE + '/'+username +'.csv'
        if not os.path.exists(profile_path):
            df_profile = self.userProfile_Generator(username)
            df_profile.to_csv(profile_path,encoding='utf-8',mode='w',header=True,index=False)
            print('User {} is a new user and the profile has been put in {}'.format(username,profile_path))
        
        else:
            df_profile = pd.read_csv(profile_path)
            df_profile.sort_values(by=['TimeStamp'],ascending= False)
            df_profile = df_profile[0:1]
        
        return df_profile

    def _updateUserCurrProfile(self,username,mode = 'auto',df_profile=None):
        profile_path = self.his.TEST_USER_PROFILE + '/'+username +'.csv'
        if not os.path.exists(profile_path):
            if mode is 'auto':
                df_profile = self.userProfile_Generator(username)
            if mode is 'manual':
                df_profile = df_profile
            df_profile.to_csv(profile_path,encoding='utf-8',mode='w',header=True,index=False)
            print('Profile Has Been Updated in ',profile_path)
            return df_profile
        
        else:
            df_profile = pd.read_csv(profile_path)
            if mode is 'auto':
                df_modified = self.oldUserProfile_Update(username,df_profile.sort_values('TimeStamp',ascending=True)[0:1])
            if mode is 'manual':
                df_modified = df_profile
            
            df_modified.to_csv(profile_path,encoding='utf-8',mode='a',header=False,index=False)
            print('Profile Has Been Updated in ',profile_path)
            return df_modified

    def userProfile_Generator(self,username):
        dataframe_lists = []

        dataframe_list = []
        userID = username
        staticProfile = self.his.staticProfile_Generator()
        dynamicProfile = self.his.dynamicProfile_Generator()

        TimeStamp = datetime.datetime.now()

        UserCurrProfile = {'TimeStamp':TimeStamp}

        UserCurrProfile.update({"User ID":userID })
        UserCurrProfile.update(staticProfile) 
        UserCurrProfile.update(dynamicProfile)

        dataframe_list.append(UserCurrProfile["TimeStamp"]) 
        dataframe_list.append(UserCurrProfile["User ID"])
        #static profile
        dataframe_list.append(UserCurrProfile['Date Of Birth'])
        dataframe_list.append(UserCurrProfile['city'])
        dataframe_list.append(UserCurrProfile['country'])
        #==========================================
        dataframe_list.append(UserCurrProfile["Loyalty Card"])
        dataframe_list.append(UserCurrProfile["Payment Card"])
        dataframe_list.append(UserCurrProfile["PRM Type"])
        dataframe_list.append(UserCurrProfile["Preferred means of transportation"])
        dataframe_list.append(UserCurrProfile["Preferred carrier"])
        dataframe_list.append(UserCurrProfile["Class"])
        dataframe_list.append(UserCurrProfile["Seat"])
        dataframe_list.append(UserCurrProfile["Refund Type"]) 

        dataframe_lists.append(dataframe_list)
        df_profile = pd.DataFrame(data=dataframe_lists)
        df_profile.columns =["TimeStamp","User ID",'Date Of Birth','city','country',
                             "Loyalty Card","Payment Card","PRM Type","Preferred means of transportation",
                             "Preferred carrier","Class","Seat","Refund Type"] 

        return df_profile

    def oldUserProfile_Update(self,username,df_old):
        dataframe_lists = []

        dataframe_list = []
        userID = username
        dynamicProfile = self.his.dynamicProfile_Generator()

        TimeStamp = datetime.datetime.now()

        UserCurrProfile = {'TimeStamp':TimeStamp}

        UserCurrProfile.update({"User ID":userID })
        UserCurrProfile.update(dynamicProfile)

        dataframe_list.append(UserCurrProfile["User ID"])
        dataframe_list.append(df_old['Date Of Birth'])
        dataframe_list.append(df_old['city'])
        dataframe_list.append(df_old['country'])
        #==========================================
        dataframe_list.append(UserCurrProfile["Loyalty Card"])
        dataframe_list.append(UserCurrProfile["Payment Card"])
        dataframe_list.append(UserCurrProfile["PRM Type"])
        dataframe_list.append(UserCurrProfile["Preferred means of transportation"])
        dataframe_list.append(UserCurrProfile["Preferred carrier"])
        dataframe_list.append(UserCurrProfile["Class"])
        dataframe_list.append(UserCurrProfile["Seat"])
        dataframe_list.append(UserCurrProfile["Refund Type"]) 

        dataframe_lists.append(dataframe_list)
        df_profile = pd.DataFrame(data=dataframe_lists)
        df_profile.columns =["TimeStamp","User ID",'Date Of Birth','city','country',
                             "Loyalty Card","Payment Card","PRM Type","Preferred means of transportation",
                             "Preferred carrier","Class","Seat","Refund Type"] 

        return df_profile

    def CHECK_USER2UPDATE_PROFILE(self,username,mode='auto',df_profile=None):
        self._updateUserCurrProfile(username,mode=mode,df_profile=df_profile)

class CLASSIFIER_RESPONSE:

    PT = ParametersTunning()
    gt_u = GET_UserProfile()
    CGY = TSP2OfferCategorizer()
    his = TEST_HistoricalDataGenerate()
    ppl = DATA_Population()

    def _get_ReqData(self,username,response_code,model_version='latest',fit_data_tag='all',normalization='zero'):

        model_name = 'best_model_{}_{}_{}'.format(username,fit_data_tag,model_version)
        model_info = self.PT.load_info(model_name)

        feature_used = model_info['feature_used']
        df_normalization = model_info['df_normalization']
        
        df_u = self.gt_u._getUserCurrProfile(username)
        offer_path = self.his.TEST_NEW_OFFER_PATH + '/' +str(response_code) + '.csv'
        df_offer =pd.read_csv(offer_path)

        if len(df_offer) >=2:
            for idx in range(len(df_offer)-1):
                df_u = pd.DataFrame.append(df_u ,df_u[0:1],ignore_index=True)
        
        df_test =pd.concat([df_u,df_offer],axis=1)
        df_raw_data = df_test.copy(deep=True) #DEEP COPY
        df_onehot = self.ppl.data_OneHot(df_raw_data)
        if len(df_onehot) > 1 :
            df_pure =self.ppl.data_Pure(df_onehot)
        else:
            df_pure = df_onehot

        df_modified,df_nouse = self.ppl.test_data_modified_on_old_model(df_pure,feature_used)
        request_data = self.ppl.data_normalization(df_normalization,df_modified,normalization)
        # request_nouse = self.data_normalization(df_nouse,normalization)
        features_nouse = df_nouse.columns.tolist()
        
        Request_dict = {'request_data':request_data,'request_nouse':df_nouse,'features_nouse':features_nouse,'request_raw_data':df_test}

        return Request_dict
        
    #########################################################################################################
    ##########                                                                                     ##########
    ##########                 USE THE SAVED MODEL TO PREDICT THE REQUEST AND GIVE ITS SCORE       ##########
    ##########                                                                                     ##########
    #########################################################################################################
    def _get_results_with_model(self,data_input,username,model_version='latest',fit_data_tag='all',model_other_path=None,info_other_path=None,checkTag=False):

        model_name = 'best_model_{}_{}_{}'.format(username,fit_data_tag,model_version)
        if checkTag is True:
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
        model = self.PT.load_model(model_name,file_path=model_other_path)
        model_info = self.PT.load_info(model_name,info_other_path)
        if checkTag is True:
            print('>>> Estimator Used is: \n>>> {}\n'.format(model_info['best_estimator']))

        res = self.PT.get_prediction(model,data_input)
        score =self.PT.get_score(model,data_input)

        if checkTag is True:
            print('>>> The predict_results of the test data is ( 0 means NOT BUY / 1 means BUY ) :\n>>> {} \n'.format(res))
            print('>>> The score of the Travel Offer can be recommended is \n>>> {}\n'.format(score))
            print('------------------------------------------------------------------------------------------\n')
        result_dict = {'predict_result':res,'travel_offer_score':score}
        return result_dict

    def API_CLASSIFIER_Response(self,username,response_code,displayNum=30,checkTag=False):
        # try:
        req = self._get_ReqData(username,response_code)
        res = self._get_results_with_model(req['request_data'],username,checkTag=checkTag)
        df_req = req['request_raw_data'].copy(deep=True)
        score = res['travel_offer_score']
        df_req['Score'] = score
        df_req = df_req.sort_values(by='Score',ascending=False)
        df_req = df_req.reset_index(drop=True)

        self.his.res2csv(df_req, self.his.TEST_RESULT_PATH, username+'_'+str(response_code)+'.csv')
        # self._Score_2_DB(res,req1,username,response_code,model_version)
        self.API_ResultDisplay(username, response_code,dispayNum=displayNum)

        # except: 
        #     print('''++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #           \n【 ERROR 】:
        #           \n>>> Data exist 
        #           \n>>> Request Score based on new model will NOT put into Recommendation Result DB for they will all be 1
        #           \n-------------------------------------------------------------------------------------------------------\n''')

    def API_ResultDisplay(self,username,response_code,dispayNum=30):

        tablename = self.his.TEST_RESULT_PATH+'/'+ username+'_'+str(response_code) + '.csv'
        df_r = pd.read_csv(tablename)
        df_res = df_r.sort_values(by='Score',ascending=False)
        df_res = df_res.reset_index(drop=True)
        
        if len(df_res) >=dispayNum:
            df_res =df_res.head(dispayNum)

        df_display = df_res[['Travel Offer ID','Starting point','Destination','Via','Departure time','Arrival time']]
        displayStringHead = '''=======================================================================================================
        \nDear {} : 
        \nAccording to your request, we recommend you consider the followiing travel offers.
        \n{}
        \n---------------------------------------------------------------------------------------------------------       
        '''.format(username,df_display)

        print(displayStringHead)

    def CHECK_USER2RESPONSE(self,username='TESTUSER1',response_code=999,req=None,Jsonfile=None):
        self.CGY.CHECK_USER2Categorizer(response_code,req,Jsonfile)
        self.gt_u._getUserCurrProfile(username)
        self.API_CLASSIFIER_Response(username,response_code,checkTag=True)

class COLD_USER_RESPONSE:

    cluster = CLUSTER()
    his = TEST_HistoricalDataGenerate()
    ppl = DATA_Population()
    gt_u =GET_UserProfile()
    classifier = CLASSIFIER_RESPONSE()


    def _insert_UserProfile(self,username,new_profile_path):
        source_filename = new_profile_path
        des_filename = self.his.TEST_USER_PROFILE + '/'+username+'.csv'
        copyfile(source_filename,des_filename)

    def _get_ColdProfileData(self,username,feature_used):
        df_raw = pd.read_csv(self.his.TEST_USER_PROFILE+'/'+username+'.csv')
        del(df_raw['TimeStamp'])
        del(df_raw['User ID'])
        df_onehot = self.ppl.data_OneHot(df_raw)
        if len(df_onehot) > 1 :
            df_pure =self.ppl.data_Pure(df_onehot)
        else:
            df_pure = df_onehot

        df_modified,df_nouse = self.ppl.test_data_modified_on_old_model(df_pure, feature_used)
        return df_modified

    #########################################################################################################
    ##########                                                                                     ##########
    ##########                    [CLUSTER]:Sign CLUSTER MODEL TO COLD USER                        ##########
    ##########                                                                                     ##########
    ######################################################################################################### 
    def API_SignColdUserModel(self,username,algorithm='DBSCAN'):
        dataInfo = self.cluster.load_info('ClusterModelINFO')
        feature_used = dataInfo['feature used']
        req = self._get_ColdProfileData(username, feature_used)
        clusterNo = self.cluster.cluster_req_predict(req, algorithm)[0]
        
        source_filename = 'best_model_Cluster_{}_Records_all_latest'.format(clusterNo)
        target_filename = 'best_model_{}_all_latest'.format(username)

        model_src = source_filename +'.m'
        model_des = target_filename +'.m'
        info_src = source_filename +'.npy'
        info_des = target_filename +'.npy'
        self.cluster.copy_model(model_src, model_des,target_path='model')
        self.cluster.copy_model(info_src, info_des,target_path='info')

        print("DONE! THE Pre trained model has been copied into the cold user") 

    #########################################################################################################
    ##########                                                                                     ##########
    ##########                    [CLUSTER]:Sign CLUSTER MODEL TO COLD USER                        ##########
    ##########                                                                                     ##########
    #########################################################################################################  

    def API_ColdUserRensponse(self,username,response_code):
        #type of profile should be dataFrame
        self.API_SignColdUserModel(username)

        self.classifier.API_CLASSIFIER_Response(username, response_code) 


    def CHECK_ColdUserResponse(self,username,response_code):
        new_profile_path = input('\n>>>please enter the path of csv document of the new user`s profile,(ENTER 0 to let the system generate a random profile for the user)')
        
        if new_profile_path == '0':
            self.gt_u._getUserCurrProfile(username)
        else:
            self._insert_UserProfile(username, new_profile_path)
        
        self.API_ColdUserRensponse(username, response_code)
        


# co = COLD_USER_RESPONSE()
# username = 'COLDUSER1'
# co.CHECK_ColdUserResponse(username,999)



# g = GET_UserProfile()
# # # g.getUserCurrProfile('TESTUSER1')
# userlist =list(map(lambda x:'CLUSTERtrain_'+str(x),range(50)))

# for username in userlist:
#     g._getUserCurrProfile(username)

# cr = CLASSIFIER_RESPONSE()
# cr.API_CLASSIFIER_Response('TESTUSER1', 999)



