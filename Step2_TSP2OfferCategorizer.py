import pandas as pd
import random

from Step1_user2TSP import USER2TSP
from LEARNER import TEST_HistoricalDataGenerate

class TSP2OfferCategorizer:

    his = TEST_HistoricalDataGenerate()

    def _OfferCategorizer(self,df_tsp,response_code):
        dataframe_lists = []
        for num in range(len(df_tsp)):
            dataframe_list = []
            category = self.his.offer_Categorizer()
            
            legs_num = df_tsp['Legs Number'][num]
            legInfo = self.his.leg_info_Generator(legs_num)


            #Category
            dataframe_list.append(category['Quick'])
            dataframe_list.append(category['Reliable'])
            dataframe_list.append(category['Cheap'])
            dataframe_list.append(category['Comfortable'])
            dataframe_list.append(category['Door-to-door'])
            dataframe_list.append(category['Environmentally friendly'])
            dataframe_list.append(category['Short'])
            dataframe_list.append(category['Multitasking'])
            dataframe_list.append(category['Social'])
            dataframe_list.append(category['Panoramic'])
            dataframe_list.append(category['Healthy'])

            #legs_info
            dataframe_list.append(legInfo['LegMode'])
            dataframe_list.append(legInfo['LegCarrier'])
            dataframe_list.append(legInfo['LegSeat'])
            dataframe_list.append(legInfo['LegLength'])

            dataframe_lists.append(dataframe_list)

        df_category = pd.DataFrame(data=dataframe_lists)
        df_category.columns=['Quick','Reliable','Cheap','Comfortable','Door-to-door',
                             'Environmentally friendly','Short','Multitasking',
                             'Social','Panoramic','Healthy','LegMode','LegCarrier','LegSeat','LegLength']
        
        df_newoffer = pd.concat([df_tsp,df_category],axis=1)

        self.his.res2csv(df_newoffer, self.his.TEST_NEW_OFFER_PATH, str(response_code) +'.csv')        
        return df_newoffer


    def CHECK_USER2Categorizer(self,response_code=999,req=None,Jsonfile=None):
        tsp = USER2TSP()
        tsp.TEST_generate_req_json(req=req)
        requestInfo = tsp.readJsonfile(Jsonfile=Jsonfile)
        df_tsp = tsp.TSP(requestInfo)
        df_newoffer = self._OfferCategorizer(df_tsp,response_code=response_code)
        self.his.res2csv(df_newoffer, tsp.CHECK_USER2TSP_PATH, 'CategorizerResult.csv')
        print('>>>\nCategorizer Phase has been finished, see the results in :\n',tsp.CHECK_USER2TSP_PATH+'CategorizerResult.csv')




# c = TSP2OfferCategorizer()
# s = USER2TSP()
# req = s.readJsonfile()
# df_tsp = s.TSP(req)
# res = c.OfferCategorizer(df_tsp)
# print(res)
# c.CHECK_USER2Categorizer()


