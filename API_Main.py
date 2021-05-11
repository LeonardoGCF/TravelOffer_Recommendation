import datetime
import random
import os
import pandas as pd

from Step1_user2TSP import USER2TSP
from Step2_TSP2OfferCategorizer import TSP2OfferCategorizer
from Step3_EnrichedOfferRanker import GET_UserProfile
from Step3_EnrichedOfferRanker import CLASSIFIER_RESPONSE
from Step3_EnrichedOfferRanker import COLD_USER_RESPONSE
from LEARNER import TEST_HistoricalDataGenerate
from LEARNER import DATA_Population
from LEARNER import CLASSIFIER_LEANER
from LEARNER import FEEDBACK
from LEARNER import CLUSTER_LEANER



class TravelOffer_RS:
    tsp = USER2TSP()
    cgy = TSP2OfferCategorizer()
    classifier = CLASSIFIER_LEANER()
    ranker = CLASSIFIER_RESPONSE()
    gt_u = GET_UserProfile()
    cluster = CLUSTER_LEANER()
    cold_ranker = COLD_USER_RESPONSE()
    fdb = FEEDBACK()
    his = TEST_HistoricalDataGenerate()

    cold2old_num = 100

    #EXAMPLE PARAMETER
    OLD_USER = 'OLDUSER1'
    COLD_USER = 'COLDUSER1'
    req =  {
                #    "Legs Number":8,
                   "Profile":'Business',
                   "Starting point":'Dublin',
                   "Destination":"Milan",
                   "Departure time":'2022-10-20 21:49',
                   "Arrival time":'2022-11-02 08:53',
                #    "Via":['Lisbon', 'Berlin', 'Kraków', 'Praha', 'Oxford', 'Madrid', 'Frankfurt'],
                   "Services":['Local type', 'Water transport', 'Telecabin'],
                #    "Transfers":"Max 3",
                #    "Transfer duration":'At least 30 min',
                #    "Walking distance to stop":'200m',
                   "Walking speed":'Slow',
                #    "Cycling distance to stop":'1800m',
                   "Cycling speed":'Fast',
                   "Driving speed":'Fast'
    }
    df_profile = pd.read_csv(his.TEST_USER_PROFILE+'/example.csv')
    

    def API_USER_TRAIN(self,username,df_profile=None,reClusterTag=True):
        userTag = self._checkColdUser(username)
        clusterExist = self._clusterModelExist()

        if not df_profile is None:
            self.gt_u._updateUserCurrProfile(username,mode='manual',df_profile=df_profile)
        elif userTag == 'new_cold_user':
            self.gt_u._updateUserCurrProfile(username)  
        
        if userTag == 'new_cold_user' or userTag == 'cold_user':
            if reClusterTag is True or clusterExist is False:
                self.cluster.update_userProfile_forCluster()
                self.cluster.API_CLUSTER_TRAINING()
                self.cold_ranker.API_SignColdUserModel(username) 
            else:
                self.cold_ranker.API_SignColdUserModel(username)
        
        if userTag == 'old_user':
            self.classifier.API_CLASSIFIER_TRAIN(username)

    def API_USER_PREDICT(self,username,req,Jsonfile=None,df_profile=None):
        userTag = self._checkColdUser(username)
        modelExist = self._classifierModelExist(username)
        response_code = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + str(random.randint(10, 99))
        if not df_profile is None:
            self.gt_u._updateUserCurrProfile(username,mode='manual',df_profile=df_profile)

        self.tsp.TEST_generate_req_json(req=req)
        requestInfo = self.tsp.readJsonfile(Jsonfile=Jsonfile)
        df_tsp = self.tsp.TSP(requestInfo)
        df_newoffer = self.cgy._OfferCategorizer(df_tsp,response_code=response_code)

        if userTag == 'new_cold_user':
            if df_profile is None:
                print('Please supply the new user`s profile info dataframe')
                return 
            clusterExist = self._clusterModelExist()
            if clusterExist == True:
                self.cold_ranker.API_ColdUserRensponse(username, response_code)
            else:
                self.API_USER_TRAIN(username,df_profile,reClusterTag=True)
                self.ranker.API_CLASSIFIER_Response(username, response_code)
            
        if userTag == 'cold_user' or userTag =='old_user':
            if modelExist == False:
                self.API_USER_TRAIN(username,df_profile,reClusterTag=False)
            self.ranker.API_CLASSIFIER_Response(username, response_code)

    def API_USER_FEEDBACK(self,username,boughtID_list,response_code):
        boughtInfo = self.fdb.get_boughtInfo(username, boughtID_list, response_code)
        self.fdb.API_UpdateRecords(username, boughtInfo, response_code)


    def CHECK_Req2TSP(self,req=None,Jsonfile=None):
        self.tsp.CHECK_USER2TSP(req,Jsonfile)

    def CHECK_Req2Categorizer(self, response_code=999,req=None,Jsonfile=None):
        self.cgy.CHECK_USER2Categorizer( response_code,req,Jsonfile)

    def CHECK_USER2UPDATE_PROFILE(self,username='OLDUSER1',mode='auto',df_profile=None):
        self.gt_u.CHECK_USER2UPDATE_PROFILE(username,mode,df_profile)

    def CHECK_USER2RESPONSE(self,username='OLDUSER1',response_code=999,req=None,Jsonfile=None):
        self.ranker.CHECK_USER2RESPONSE(username,response_code,req,Jsonfile)

    def CHECK_ColdUserResponse(self,username='COLDUSER1',response_code=999,req=None,Jsonfile=None):
        self.cgy.CHECK_USER2Categorizer(response_code,req,Jsonfile)
        self.cold_ranker.CHECK_ColdUserResponse(username, response_code)

    def CHECK_USER_FEEDBACK(self,username='TESTUSER1',boughtID_list=[2021043019424627],response_code=999):
        # boughtID_list=input('Please enter the travel offer id list that the user')
        self.fed.CHECK_USER_FEEDBACK(username,boughtID_list,response_code)

    def CHECK_ADMINISTRATOR2CLASSIFIER_TRAIN(self,username='OLDUSER1'):
        self.classifier.CHECK_ADMINISTRATOR2CLASSIFIER_TRAIN(username)

    def CHECK_ADMINISTRATOR2CLUSTER_TRAIN(self,userlist=None):
        if userlist is None:
            userlist =list(map(lambda x:'CLUSTERtrain_'+str(x),range(50)))

        for username in userlist:
            self.gt_u._getUserCurrProfile(username)
            self.his._GenerateHis2File(username)
        self.cluster.CHECK_ADMINISTRATOR2CLUSTER_TRAIN()


    def _checkColdUser(self,username):
        his_file = self.his.TEST_HISTORICAL_DATA_PATH
        if not username+'.csv' in os.listdir(his_file):
            userTag = 'new_cold_user'
        elif len(pd.read_csv(his_file+'/'+username+'.csv')) < self.cold2old_num:
            userTag = 'cold_user'
        else:
            userTag = 'old_user'

        return userTag

    def _clusterModelExist(self):
        
        if 'DBSCAN_newest_cluster_model.m' in os.listdir(self.his.CM_FOLDER_PATH ):
            clusterExist = True
        else:
            clusterExist = False
        return clusterExist

    def _classifierModelExist(self,username):
        if 'best_model_{}_all_latest.m'.format(username) in os.listdir(self.his.MODEL_FOLDER_PATH):
            classifierExist = True
        else:
            classifierExist = False
        return classifierExist

    def show(self):
        show_body = '''\n==================================================================
                       \n API LIST:
                       \n
                       \n>>>"API_USER_TRAIN":
                       \n       -used by the administrator to train or update the model for user
                       \n
                       \n>>>"API_USER_PREDICT":
                       \n       -used by the user to get the recommendation list
                       \n
                       \n>>>"API_USER_FEEDBACK":
                       \n       -used by the administrator to update the historical records according to the user's decision
                       \n
                       \n----------------------------------------
                       \n CHECK LIST:
                       \n
                       \n>>>"CHECK_Req2TSP":
                       \n       -check the phase from user to tsp 
                       \n        
                       \n>>>"CHECK_Req2Categorizer":
                       \n       -check the pahse from user to categorizer
                       \n
                       \n>>>"CHECK_USER2UPDATE_PROFILE":
                       \n       -check the functionality that the user want to update his/her profile
                       \n
                       \n>>>"CHECK_USER2RESPONSE":
                       \n       -check the predict function for an old user
                       \n
                       \n>>>"CHECK_ColdUserResponse":
                       \n       -check the predict function for a cold user
                       \n
                       \n>>>"CHECK_USER_FEEDBACK":
                       \n       -check the feedback function 
                       \n
                       \n>>>"CHECK_ADMINISTRATOR2CLASSIFIER_TRAIN":
                       \n       -check the functionality that the administrator want to train/update the classifier model
                       \n
                       \n>>>"CHECK_ADMINISTRATOR2CLUSTER_TRAIN":
                       \n       -check the functionality that the administrator want to train/update the cluster model
                       \n
                       \n -----------------------------------------
                       \n [Using help() to check the details of the function ]
                    '''
        print(show_body)

    def help(self):
        help_code = input('Please enter the func name that you want to know the details: ')

        # if help_code is 
        if help_code == "CHECK_Req2TSP":

            content = '''\n==================================================================
                         \nCHECK_Req2TSP(req=None,Jsonfile=None) :
                         \n 
                         \nif req & Jsonefile is None, func will use the default req dict = TEST_search_option_dict
                         \nif user gives the req dict, func will use the req dict user gives and generate responding Jsonefile.
                         \nif user gives the Jsonefile path,func will use the req info in Jsonefile.
                         \nif user gives req dict and Jsonefile path,func will use the req info in Jsonefile.
                      '''
            print(content)

        if help_code == "CHECK_Req2Categorizer":

            content = '''\n==================================================================
                         \nCHECK_Req2Categorizer(response_code=999,req=None,Jsonfile=None) :
                         \n 
                         \n[response_code] will be set as the travel offer`s tablename that supplied by third parties.
                         \n[req] is a dict that contains the request of the user, if NONE req will be the default request dict.
                         \n[Jsonfile] is a path of the request Jsonfile, if given system will use the req in Jsonfile.
                         \n
                      '''
            print(content)

        if help_code == "CHECK_USER2UPDATE_PROFILE":

            content = '''\n==================================================================
                         \nCHECK_USER2UPDATE_PROFILE(username='OLDUSER1',mode='auto',df_profile=None):
                         \n 
                         \n[username] name of the user that you want to update profile, if not given username will be 'OLDUSER1'
                         \n[mode] update mode. 'auto': randomly generate the new profile info. 'manual': use df_profile to update the user
                         \n[df_profile] NONE if mode is 'auto; given by the user when mode is 'manual' 
                         \n
                      '''
            print(content)

        if help_code == "CHECK_USER2RESPONSE":

            content = '''\n==================================================================
                         \nCHECK_USER2RESPONSE(username='OLDUSER1',response_code=999,req=None,Jsonfile=None):
                         \n 
                         \n[username] name of the user that you want to predict, if not given username will be 'OLDUSER1'
                         \n[response_code] used to find the travel offer list satisfied the request. If not given, list will be in 999.csv.
                         \n[req] is a dict that contains the request of the user, if NONE req will be the default request dict.
                         \n[Jsonfile] is a path of the request Jsonfile, if given system will use the req in Jsonfile.
                         \n
                      '''
            print(content)

        if help_code == "CHECK_ColdUserResponse":

            content = '''\n==================================================================
                         \nCHECK_ColdUserResponse(username='COLDUSER1',response_code=999,req=None,Jsonfile=None):
                         \n 
                         \n[username] name of the user that you want to predict, if not given username will be 'COLDUSER1'
                         \n[response_code] used to find the travel offer list satisfied the request. If not given, list will be in 999.csv.
                         \n[req] is a dict that contains the request of the user, if NONE req will be the default request dict.
                         \n[Jsonfile] is a path of the request Jsonfile, if given system will use the req in Jsonfile.
                         \n
                      '''
            print(content)

        if help_code == "CHECK_USER_FEEDBACK":

            content = '''\n==================================================================
                         \nCHECK_USER_FEEDBACK(username='TESTUSER1',boughtID_list=[2021043019424627],response_code=999):
                         \n 
                         \n[username] name of the user who bought the offer, if not given username will be 'TESTUSER1'
                         \n[boughtID_list] a list contains all the travel offer ID that the user has bought already.
                         \n[response_code] used to find the travel offer list satisfied the request. If not given, list will be in 999.csv.
                         \n
                      '''
            print(content)

        if help_code == "CHECK_ADMINISTRATOR2CLASSIFIER_TRAIN":

            content = '''\n==================================================================
                         \nCHECK_ADMINISTRATOR2CLASSIFIER_TRAIN(username='OLDUSER1'):
                         \n 
                         \n[username] name of the user whoes model will be updated , if not given username will be 'OLDUSER1'
                         \n
                      '''
            print(content)

        if help_code == "CHECK_ADMINISTRATOR2CLUSTER_TRAIN":

            content = '''\n==================================================================
                         \nCHECK_ADMINISTRATOR2CLUSTER_TRAIN(userlist=None):
                         \n 
                         \n[userlist] NONE means system will automatically generate 50 CLUSTERtrain_n user profile info.  
                         \nGiven a list of usernames, system will randomly generate all the responding profile and historical records
                         \nand add into the files.
                      '''
            print(content)

        if help_code == "API_USER_TRAIN":

            content = '''\n==================================================================
                         \nAPI_USER_TRAIN(username,df_profile=None,reClusterTag=True)
                         \n 
                         \n[username] the person for whom you try to train the model, can be an new_cold_user with a random name.
                         \n           can be a cold_user whoes #historical data less than 20,
                         \n           can be an old_user ,like 'OLDUSER1'
                         \n[df_profile] If exist, update user profile with this data.
                         \n[reClusterTag] If True, update the CLUSTER MODEL. Use old CLUSTER MODEL if False
                      '''
            print(content)
            
        if help_code == "API_USER_PREDICT":

            content = '''\n==================================================================
                         \nAPI_USER_PREDICT(username,req,Jsonfile=None,df_profile=None)
                         \n 
                         \n[username] the person for whom you try to predict, can be an new_cold_user with a random name.
                         \n           can be a cold_user whoes #historical data less than 20,
                         \n           can be an old_user ,like 'OLDUSER1'
                         \n[req] is a dict that contains the request of the user, if NONE req will be the default request dict==> .tsp.TEST_search_option_dict.
                         \n[Jsonfile] is a path of the request Jsonfile, if given system will use the req in Jsonfile.
                         \n[df_profile] If exist, update user profile with this data.
                         \n[reClusterTag] Always set as False
                         \n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                         \n You can use the following default parameters : 
                         \n [username] self.OLD_USER ==> an old user / self.COLD_USER ==> an cold user
                         \n [req] self.req
                         \n [df_profile] self.df_profile , do changes like 'df_profile.loc[0,'User ID] = username' before using
                      '''
            print(content)

        if help_code == "API_USER_FEEDBACK":

            content = '''\n==================================================================
                         \nAPI_USER_FEEDBACK(self,username,boughtID_list,response_code):
                         \n 
                         \n[username] name of the user who bought the offer, if not given username will be 'TESTUSER1'
                         \n[boughtID_list] a list contains all the travel offer ID that the user has bought already.
                         \n[response_code] used to find the travel offer list satisfied the request. If not given, list will be in 999.csv.
                         \n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                         \n[username] self.OLD_USER ,try to not use self.COLD_USER or it will make the cold user to be an old user.
                         \n[boughtID_list] enter by the user.
                      '''
            print(content)

        else:
            print('{} not in the help list, use show() to check all the function'.format(help_code))
            

# main = TravelOffer_RS()

# main.help()
# username = 'COLDUSER3'
# username = 'COLDUSER1'
# username = 'OLDUSER1'
# main.API_USER_TRAIN(username,reClusterTag=False)
# main.show()





