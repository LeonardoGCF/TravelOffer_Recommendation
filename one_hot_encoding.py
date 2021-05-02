import numpy
import pandas as pd
# import sklearn
# from sklearn.preprocessing import OneHotEncoder

class OfferOneHotEncoder:
    def read_Data(self):
        self.FILE_PATH = './Meeting1/travel_offer.csv'
        df = pd.read_csv(self.FILE_PATH)
        return df

    def list2number(self,df,ObjCol,strand_list):
        for num in range(len(df[ObjCol])):
            list_ =df[ObjCol][num]
            list_tmp = eval(list_)
            # strand_list = ["Cartafreccia","FlyingBlue","Golden Card","Grand Voyageur"]
            output_list = [0]*len(strand_list)
            for inx in range(len(list_tmp)):
                if list_tmp[inx] in strand_list:
                    tag = strand_list.index(list_tmp[inx])
                    output_list[tag] +=1
                else:
                    print("WRONG LIST")
            # print(output_list)
            df[ObjCol][num] = output_list
        return df[ObjCol] 

    def listColSplit(self,df,ObjCol,strand_list):
        for inx in range(len(strand_list)):
            strand_list[inx] = ObjCol +'-'+ strand_list[inx]
        df_tmp = df[ObjCol].apply(pd.Series,index=strand_list)
        df = pd.concat([df,df_tmp],axis=1)
        del df[ObjCol]
        return df

    def str2list(self,df,ObjCol):
        for inx in range(len(df[ObjCol])):
            df[ObjCol][inx] = eval(df[ObjCol][inx])

        return df[ObjCol]

    def one_hot_encoder(self,df,ObjCol):
        df_tmp = pd.get_dummies(df[ObjCol])
        df = pd.concat([df,df_tmp],axis=1)
        del df[ObjCol]
        return df

    def df_Regenerator(self,df):

        #NULL==>0
        # df.fillna(value=0)
        #DoB
        col = df.columns.values.tolist()

        if 'Date Of Birth' in col:
            df['Date Of Birth'] = pd.to_datetime(df['Date Of Birth'],format='%Y-%m-%d')
            df['DoB-year']=df['Date Of Birth'].dt.year
            df['DoB-month']=df['Date Of Birth'].dt.month
            df['DoB-day']=df['Date Of Birth'].dt.day
            del df['Date Of Birth']
        #City&country
        # df = pd.get_dummies(df['city'])
        # df = pd.get_dummies(df['country'])
        #Card
        obj1 = 'Loyalty Card'
        if obj1 in col:
            list1 = ["Cartafreccia","FlyingBlue","Golden Card","Grand Voyageur"]
            self.list2number(df,obj1,list1)
            df = self.listColSplit(df,obj1,list1) 
        #Payment Card
        obj2 = 'Payment Card'
        if obj2 in col:
            list2 = ["Mastercard","Visa","Paypal","Google Wallet","Apple Wallet"]
            self.list2number(df,obj2,list2)
            df = self.listColSplit(df,obj2,list2)
        #PRM types
        obj3 = 'PRM Type'
        if obj3 in col:
            list3 = ["Older person",
                    "Persons with impairments in their members / users of temporary wheelchair",
                    "Persons porting a carrycots",
                    "Persons with blind or visual impairments",
                    "Wheelchair users in mainstreaming seat",
                    "Wheelchair users in specific seat named h-seat",
                    "Pregnant woman",
                    "Person with deafness or auditory impairments"]
            self.list2number(df,obj3,list3)
            df = self.listColSplit(df,obj3,list3)
        #Preferred transportation
        obj4 ='Preferred means of transportation'
        if obj4 in col:
            list4 = ["Coach","Toll","Car Sharing","Train","Airline",
                        "Urban","Trolely Bus","Tram","Intercity","Metro",
                        "Ship","Cable Way","Funicular","Taxi","Bus","Other",
                        "Park","Bike Sharing"]
            self.list2number(df,obj4,list4)
            df = self.listColSplit(df,obj4,list4)
        #Preferred carrier
        obj5 = 'Preferred carrier'
        if obj5 in col:
            list5 = ["Trenitalia","SNFC","AirFrance","VBB","TMB","Renfe","RegioJet","KLM","Iberia","FlixBus"]
            self.str2list(df,obj5)
            df = self.listColSplit(df,obj5,list5)
        #class seat refund type
        # df = pd.get_dummies(df['Class'])
        # df = pd.get_dummies(df['Seat'])
        # df = pd.get_dummies(df['Refund Type'])
        #category KEEP
        #legsnumber KEEp
        #profile
        # df = pd.get_dummies(df['Profile'])
        # df = pd.get_dummies(df['Starting point'])
        # df = pd.get_dummies(df['Destination'])
        #via
        obj6 ='Via'
        if obj6 in col:
            list6 = ["Brugge","Bruxelle","Praha","Krumlov","Paris","Nice",
                        "Frankfurt","Berlin","Dublin","Crok","Milan","Rome",
                        "Warsaw","Kraków","Lisbon","Porto","Barcelona","Madrid",
                        "London","Oxford"]
            self.list2number(df,obj6,list6)
            df = self.listColSplit(df,obj6,list6)
        #Leg
        obj7 = 'LegMode'
        if obj7 in col:
            list7 =["Coach","Toll","Car Sharing","Train","Airline",
                        "Urban","Trolely Bus","Tram","Intercity","Metro",
                        "Ship","Cable Way","Funicular","Taxi","Bus","Other",
                        "Park","Bike Sharing"]
            self.list2number(df,obj7,list7)
            df = self.listColSplit(df,obj7,list7)
        # print(df['LegMode-Car Sharing'])
        obj8 ='LegCarrier'
        if obj8 in col:
            list8 = ["Trenitalia","SNFC","AirFrance","VBB","TMB","Renfe","RegioJet","KLM","Iberia","FlixBus"]
            self.list2number(df,obj8,list8)
            df = self.listColSplit(df,obj8,list8)
        #
        obj9 = 'LegSeat'
        if obj9 in col:
            list9 = ["Aisle","Window","Large"]
            self.list2number(df,obj9,list9)
            df = self.listColSplit(df,obj9,list9)
        #length ===>DROP TODO
        if 'LegLength' in col:
            del(df['LegLength'])

        # Departure time &Arrival time
        if 'Departure time' in col:
            df['Departure time'] = pd.to_datetime(df['Departure time'],format='%Y-%m-%d %H:%M:%S')
            df['DT-year']=df['Departure time'].dt.year
            df['DT-month']=df['Departure time'].dt.month
            df['DT-day']=df['Departure time'].dt.day
            df['DT-Hour'] = df['Departure time'].dt.hour
            df['DT-Minute'] = df['Departure time'].dt.minute
            del df['Departure time']

        if 'Arrival time' in col:
            df['Arrival time'] = pd.to_datetime(df['Arrival time'],format='%Y-%m-%d %H:%M:%S')
            df['AT-year']=df['Arrival time'].dt.year
            df['AT-month']=df['Arrival time'].dt.month
            df['AT-day']=df['Arrival time'].dt.day
            df['AT-Hour'] = df['Arrival time'].dt.hour
            df['AT-Minute'] = df['Arrival time'].dt.minute
            del df['Arrival time']

        #Services
        obj10 ='Services'
        if obj10 in col:
            list10 = ["Air","Highspeed train","Train","Coach","Local rail",
                    "Suburban","Underground","Tram","Bus","DRT","Water transport",
                    "Telecabin","Local type","Miscellaneous"]
            self.list2number(df,obj10,list10)
            df = self.listColSplit(df,obj10,list10)

        #Transfers

        df = pd.get_dummies(df)
        return df

    def res2csv(self,df,target_path):
        df.to_csv(target_path,encoding='utf-8',mode='w',header=True,index=False) 



# encoder = OfferOneHotEncoder()
# df = encoder.read_Data()
# df = encoder.df_Regenerator(df)
# tp = './Meeting1/one_hot_encoded.csv'
# encoder.res2csv(df,tp)
