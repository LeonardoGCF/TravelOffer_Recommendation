import json
import os
import datetime
import time
import random
import pandas as pd

from LEARNER import TEST_HistoricalDataGenerate

class USER2TSP:
    TEST_search_option_dict = {
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
    his = TEST_HistoricalDataGenerate()

    FILE_PATH = os.path.dirname(os.path.abspath(__file__))
    TEST_DATA_PATH = os.path.abspath(os.path.join(FILE_PATH,'TEST_DATA'))
    TEST_REQUEST_PATH = os.path.abspath(os.path.join(TEST_DATA_PATH,'RequestJson'))
    CHECK_USER2TSP_PATH = os.path.abspath(os.path.join(TEST_DATA_PATH,'CHECK_Metadata'))

    def TEST_generate_req_json(self,req=None):
        if req is None:
            req = self.TEST_search_option_dict

        if not os.path.exists(self.TEST_REQUEST_PATH):
            os.mkdir(self.TEST_REQUEST_PATH)
        Jsonfile = self.TEST_REQUEST_PATH + '/requestInfo.json'

        with open(Jsonfile,'w') as f :
            json.dump(req,f,indent=4)
            print('\n>>>\nThe request of TEST USER has been generated in \n',Jsonfile )

    def readJsonfile(self,Jsonfile=None):
        if Jsonfile is None:
            Jsonfile = self.TEST_REQUEST_PATH+'/requestInfo.json'
        with open(Jsonfile,'r') as f:
            requestInfo = json.load(f)
        return requestInfo


    def TSP(self,requestInfo):
        self.his.dataRange_OfferContext()

        dataframe_lists = []
        unique_starter = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        TravelOfferNum = random.randint(20,50)

        travelOfferID_list = list(map(lambda x:str(unique_starter)+str(x),range(TravelOfferNum)))

        for travelOfferID in travelOfferID_list:
            dataframe_list = []
            legs_num = random.randint(1,10)

            offerContext = self.his.offerContext_Generator(legs_num)
            offerContext.update(requestInfo)


            NewOffer ={"Travel Offer ID":travelOfferID}

            NewOffer.update(offerContext)

            dataframe_list.append(NewOffer["Travel Offer ID"])
            #legs_num
            dataframe_list.append(NewOffer['Legs Number'])
            #profile
            dataframe_list.append(NewOffer['Profile'])
            #Source-Des-Via
            dataframe_list.append(NewOffer['Starting point'])
            dataframe_list.append(NewOffer['Destination'])
            dataframe_list.append(NewOffer['Via'])

            #context
            dataframe_list.append(NewOffer["Departure time"])
            dataframe_list.append(NewOffer["Arrival time"])
            dataframe_list.append(NewOffer["Services"])
            dataframe_list.append(NewOffer["Transfers"])
            dataframe_list.append(NewOffer["Transfer duration"])
            dataframe_list.append(NewOffer["Walking distance to stop"])
            dataframe_list.append(NewOffer["Walking speed"])
            dataframe_list.append(NewOffer["Cycling distance to stop"])
            dataframe_list.append(NewOffer["Cycling speed"])
            dataframe_list.append(NewOffer["Driving speed"])
            # print(dataframe_list)
            dataframe_lists.append(dataframe_list)

        df_newoffer = pd.DataFrame(data=dataframe_lists)
        df_newoffer.columns =["Travel Offer ID",'Legs Number','Profile','Starting point',
                     'Destination','Via',"Departure time",
                     "Arrival time","Services","Transfers","Transfer duration","Walking distance to stop",
                     "Walking speed","Cycling distance to stop","Cycling speed","Driving speed"] 

        return df_newoffer
        

    def CHECK_USER2TSP(self,req=None,Jsonfile=None):
        self.TEST_generate_req_json(req=req)
        requestInfo = self.readJsonfile(Jsonfile=Jsonfile)
        df_tsp = self.TSP(requestInfo)
        self.his.res2csv(df_tsp, self.CHECK_USER2TSP_PATH, 'TSPresults.csv')
        print('>>>\nTSP PHASE has been checked over, see the results in :\n',self.CHECK_USER2TSP_PATH+'/TSPresults.csv')



# s = USER2TSP()
# s.CHECK_USER2TSP()
# requestInfo = s.readJsonfile()
# s.TSP(requestInfo)
# df = s.TSP(requestInfo)
# print(df)
# s.TEST_generate_req_json()
# print(s.readJsonfile())

'''
'Quick','Reliable','Cheap','Comfortable','Door-to-door','Environmentally friendly','Short','Multitasking',
                     'Social','Panoramic','Healthy',
'''

