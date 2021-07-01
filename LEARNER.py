import time
import random
import datetime
import pandas as pd
import numpy as np
import os


from sklearn import preprocessing

from one_hot_encoding import OfferOneHotEncoder
from ParametersTunning import ParametersTunning
from Cluster import CLUSTER

import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns



class TEST_HistoricalDataGenerate:

    FILE_PATH = os.path.dirname(os.path.abspath(__file__))
    TEST_DATA_PATH = os.path.abspath(os.path.join(FILE_PATH,'TEST_DATA'))
    TEST_HISTORICAL_DATA_PATH = os.path.abspath(os.path.join(TEST_DATA_PATH,'HISTORICAL_DATA'))
    TEST_NEW_OFFER_PATH = os.path.abspath(os.path.join(TEST_DATA_PATH,'NEW_OFFER_LIST')) 
    TEST_RESULT_PATH = os.path.abspath(os.path.join(TEST_DATA_PATH,'RECOMMENDATION_RESULTS'))
    CHECK_USER2TSP_PATH = os.path.abspath(os.path.join(TEST_DATA_PATH,'CHECK_Metadata'))
    TEST_USER_PROFILE = os.path.abspath(os.path.join(TEST_DATA_PATH,'USER_PROFILE'))
    EVAL_ACC_RESULT_PATH =os.path.abspath(os.path.join(TEST_DATA_PATH,'EVAL_ACC_RESULT'))

    CLUSTER_FOLDER_PATH =os.path.abspath(os.path.join(TEST_DATA_PATH,'CLUSTER_FOLDER'))
    CM_FOLDER_PATH = os.path.abspath(os.path.join( CLUSTER_FOLDER_PATH,'CLUSTER_MODEL_INFO'))

    CLASSIFIER_PATH = os.path.abspath(os.path.join(TEST_DATA_PATH,'CLASSIFIER'))
    MODEL_FOLDER_PATH = os.path.abspath(os.path.join(CLASSIFIER_PATH,'MODEL'))

    def dataRange_StaticProfile(self):
        #Contact Information
        self.countries = ["Belgium","Czechia","France","Germany","Ireland",
                   "Italy","Poland","Portugal","Spain","UK"]
    
        self.city_for_Belgium = ["Brugge","Bruxelle"]
        self.city_for_Czechia = ["Praha","Krumlov"]
        self.city_for_France = ["Paris","Nice"]
        self.city_for_Germany = ["Frankfurt","Berlin"]
        self.city_for_Ireland = ["Dublin","Crok"]
        self.city_for_Italy = ["Milan","Rome"]
        self.city_for_Poland = ["Warsaw","Kraków"]
        self.city_for_Portugal = ["Lisbon","Porto"]
        self.city_for_Spain =["Barcelona","Madrid"]
        self.city_for_UK = ["London","Oxford"]

        self.cities = ["Brugge","Bruxelle","Praha","Krumlov","Paris","Nice",
                       "Frankfurt","Berlin","Dublin","Crok","Milan","Rome",
                       "Warsaw","Kraków","Lisbon","Porto","Barcelona","Madrid",
                       "London","Oxford"]

    def dataRange_DynamicProfile(self):
        #profile
        self.loyalty_cards = ["Cartafreccia","FlyingBlue","Golden Card","Grand Voyageur"]
        self.paymant_cards = ["Mastercard","Visa","Paypal","Google Wallet","Apple Wallet"]
        self.prm_types = ["Older person",
                "Persons with impairments in their members / users of temporary wheelchair",
                "Persons porting a carrycots",
                "Persons with blind or visual impairments",
                "Wheelchair users in mainstreaming seat",
                "Wheelchair users in specific seat named h-seat",
                "Pregnant woman",
                "Person with deafness or auditory impairments"]
    
        #journey
        self.transportations = ["Coach","Toll","Car Sharing","Train","Airline",
                       "Urban","Trolely Bus","Tram","Intercity","Metro",
                       "Ship","Cable Way","Funicular","Taxi","Bus","Other",
                       "Park","Bike Sharing"]

        self.carriers =["Trenitalia","SNFC","AirFrance","VBB","TMB","Renfe","RegioJet","KLM","Iberia","FlixBus"]

        self.class_types = ["Economy","Business","First Class"]
        self.seats = ["Aisle","Window","Large"]
        self.refund_types = ["Automatic refund","Manual refund"]
   
    def dataRange_OfferContext(self):
        self.cities = ["Brugge","Bruxelle","Praha","Krumlov","Paris","Nice",
                       "Frankfurt","Berlin","Dublin","Crok","Milan","Rome",
                       "Warsaw","Kraków","Lisbon","Porto","Barcelona","Madrid",
                       "London","Oxford"]
        self.profile_typies = ["Basic","Business","Family","Leisure"]
        self.starting_points = self.cities
        self.destination_points = self.cities
        departure_time = None
        arrival_time = None
        via_stop = None
        self.services = ["Air","Highspeed train","Train","Coach","Local rail",
                "Suburban","Underground","Tram","Bus","DRT","Water transport",
                "Telecabin","Local type","Miscellaneous"]

        self.transfers = ["Unlimited","None","Max 1","Max 2","Max 3","Max 4"]

        self.transfer_durations = ["normal","At least 10 min","At least 15 min","At least 20 min",
                          "At least 25 min","At least 30 min","At least 35 min",
                          "At least 40 min","At least 45 min"]
            
        self.walking_speads = ["Slow","Medium","Fast"]

        self.cycling_speads = ["Slow","Medium","Fast"]

        self.driving_speeds = ["Slow","Medium","Fast"]

    def dataRange_OfferCategory(self):
        self.Categories = ["Quick", "Reliable", "Cheap", "Comfortable", "Door-to-door", "Environmentally friendly",
              "Short", "Multitasking", "Social", "Panoramic", "Healthy"]
  
    def dataRang_LegInfo(self):
        self.Leg_Modes = ["Coach","Toll","Car Sharing","Train","Airline",
                       "Urban","Trolely Bus","Tram","Intercity","Metro",
                       "Ship","Cable Way","Funicular","Taxi","Bus","Other",
                       "Park","Bike Sharing"]

        self.Leg_Carriers = ["Trenitalia","SNFC","AirFrance","VBB","TMB","Renfe","RegioJet","KLM","Iberia","FlixBus"]

        self.Leg_Seats = ["Aisle","Window","Large"]


    def offer_Categorizer(self):
        self.dataRange_OfferCategory()
        OfferScore = {}

        for i in self.Categories :
         score = random.uniform(0,1)
         OfferScore.update({i:round(score,4)})

        return OfferScore

    def leg_info_Generator(self,legs_num):
        self.dataRang_LegInfo()
        leg_mode = []
        leg_carrier = []
        leg_seat = []
        leg_length = []

        for inx in range(legs_num):
            leg_mode.append(random.choice(self.Leg_Modes))
            leg_carrier.append(random.choice(self.Leg_Carriers)) 
            leg_seat.append(random.choice(self.Leg_Seats))
            leg_length.append(random.randint(10,100))

        leg_tmp = {"LegMode":str(leg_mode),"LegCarrier":str(leg_carrier),"LegSeat":str(leg_seat),"LegLength":str(leg_length)}
        
        return leg_tmp

    #--------RANDOM--------------------
    def staticProfile_Generator(self):
        self.dataRange_StaticProfile()
        #personal Information-date of birth
        start_birth =(1920,1,1,10,10,10,10,10,10) 
        end_birth = time.localtime()
        start = time.mktime(start_birth)
        end =time.mktime(end_birth)
        s =random.randint(start,end)
        date_str = datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=s)
        date_of_birth = date_str.strftime("%Y-%m-%d")
        # print(date_of_birth)

        #country
        country = random.choice(self.countries)
        # print(country)

        #city

        city_search_name = "self.city_for_"+country
        city = random.choice(eval(city_search_name))
        # print(city)

        # if country is "UK":
        #     country = "United Kingdom"
        static_profile_dict ={"Date Of Birth":date_of_birth,"city":city,"country":country}
        # print("\nRandom Registration Information has been generated:\n")
        # print(self.registration_dict)
        return static_profile_dict

    def dynamicProfile_Generator(self):
        self.dataRange_DynamicProfile()
        #cards
        num_of_cards = random.randint(0,4)
        loyalty_card = random.sample(self.loyalty_cards,num_of_cards)
        # print(num_of_cards)
        # print(loyalty_card)

        num_of_paymentcards = random.randint(0,5)
        payment_card = random.sample(self.paymant_cards,num_of_paymentcards)
        # print(payment_card)

        num_of_PRM = random.randint(0,8)
        PRM_type = random.sample(self.prm_types,num_of_PRM)
        # print(PRM_type)

        #journey
        num_of_transportation = random.randint(1,18)
        preferred_transportation =random.sample(self.transportations,num_of_transportation)
        # print(preferred_transportation)

        #preferred carrier
        preferred_carrier = []
        for i in range(len(self.carriers)):
            preferred_carrier.append(random.randint(0,10)/2)
        # print(preferred_carrier)

        #class
        class_type = random.choice(self.class_types)
        # print(class_type)

        #seat
        seat_type = random.choice(self.seats)
        # print(seat_type)

        #refund
        refund_type = random.choice(self.refund_types)
        # print(refund_type)

        #all
        dynamic_profile_dict = {"Loyalty Card":str(loyalty_card),
                           "Payment Card":str(payment_card),
                           "PRM Type":str(PRM_type),
                           "Preferred means of transportation":str(preferred_transportation),
                           "Preferred carrier":str(preferred_carrier),
                           "Class":class_type,
                           "Seat":seat_type,
                           "Refund Type":refund_type}

        # print("\nRandom Preference Information has been generated:\n")
        # print(self.preference_dict)
        return dynamic_profile_dict

    def offerContext_Generator(self,legs_num):
        self.dataRange_OfferContext()
        #profile
        # profile_type = random.sample(self.profile_typies,1)
        profile_type =random.choice(self.profile_typies)
        # print(profile_type)


        # start_point = self.registration_dict["city"]
        start_point = random.choice(self.starting_points)
        # print(start_point)
        
        destination = random.choice(self.destination_points)
        # print(destination)

        #departure_time
        start_departure = time.localtime() 
        end_departure =(2022,12,31,0,0,0,0,0,0) 
        start_dep = time.mktime(start_departure)
        end_dep =time.mktime(end_departure)
        sec_dep =random.randint(start_dep,end_dep)
        date_dep = datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=sec_dep)
        departure_time = date_dep.strftime("%Y-%m-%d %H:%M")
        # print(departure_time)

        #arrival time
        start_arr = sec_dep
        end_arrival = (2022,12,31,12,12,0,0,0,0)
        end_arr = time.mktime(end_arrival)
        sec_arr = random.randint(start_arr,end_arr)
        date_arr = datetime.datetime(1970,1,1) + datetime.timedelta(seconds=sec_arr)
        arrival_time = date_arr.strftime("%Y-%m-%d %H:%M") 
        # print(arrival_time)
        
        #via
        # num_of_via = random.randint(0,8)
        # via_dict=self.countries
        # via_dict.remove(self.registration_dict['country'])
        # if destination != self.registration_dict["country"]:
            # via_dict.remove(destination)
        num_of_via = legs_num -1
        via_dict = ["Brugge","Bruxelle","Praha","Krumlov","Paris","Nice",
                       "Frankfurt","Berlin","Dublin","Crok","Milan","Rome",
                       "Warsaw","Kraków","Lisbon","Porto","Barcelona","Madrid",
                       "London","Oxford"]

        via_dict.remove(start_point)
        if destination  != start_point:
            via_dict.remove(destination) 
        via_stop = random.sample(via_dict,num_of_via)
        

        #services
        num_of_services =random.randint(0,14)
        service =random.sample(self.services,num_of_services)
        # print(service)

        #transfers
        transfer = random.choice(self.transfers)
        # print(transfer)

        #transfer duration
        transfer_duration = random.choice(self.transfer_durations)
        # print(transfer_duration)

        #walking distance
        walking_distance_to_stop = str(random.randrange(500,5000,500))+'m'
        # print(walking_distance_to_stop)

        #walking speed
        walking_speed = random.choice(self.walking_speads)
        # print(walking_speed)

        cycling_distance_to_stop = str(random.randrange(2000,20000,500))+'m'
        # print(cycling_distance_to_stop)

        cycling_speed = random.choice(self.cycling_speads)
        # print(cycling_speed)

        driving_speed = random.choice(self.driving_speeds)
        # print(driving_speed)

        # print("\nRandom Trip Planner Information has been generated:\n")
        OfferContext_dict = {"Legs Number":legs_num,"Profile":profile_type,
                   "Starting point":start_point,"Destination":destination,
                   "Departure time":departure_time,"Arrival time":arrival_time,
                   "Via":str(via_stop),"Services":str(service),"Transfers":transfer,
                   "Transfer duration":transfer_duration,
                   "Walking distance to stop":walking_distance_to_stop,
                   "Walking speed":walking_speed,
                   "Cycling distance to stop":cycling_distance_to_stop,
                   "Cycling speed":cycling_speed,
                   "Driving speed":driving_speed}

        # print(TP_dict)
        return OfferContext_dict

    def historical_Generator(self,username,staticChangeTag=True,staProfile=None):

        dataframe_lists = []
        unique_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S')      
        
        userID = username
        if staticChangeTag is True:
            staticProfile = self.staticProfile_Generator()
        else:
            staticProfile = staProfile
            if staProfile is None:return
        dynamicProfile = self.dynamicProfile_Generator()

        TimeStamp = datetime.datetime.now()

        TravelOffer = {"TimeStamp":TimeStamp}
        TravelOffer.update({"User ID":userID })
        TravelOffer.update(staticProfile)
            
        #change the number to generate different amount of his records
        dynamic_num =random.randint(4,7)
        for inx_d in range(dynamic_num):
            TravelOffer.update(dynamicProfile)
                
            travel_num =random.randint(30,50)
            for n in range(travel_num):
                dataframe_list = []
                travelOfferID = str(unique_id) + str(n)
                legs_num = random.randint(1,10)
                # print(legs_num)

                category = self.offer_Categorizer()
                offerContext = self.offerContext_Generator(legs_num)
                legInfo = self.leg_info_Generator(legs_num)

                TravelOffer.update({"Travel Offer ID":travelOfferID})
                TravelOffer.update(category)
                TravelOffer.update(legInfo)
                TravelOffer.update(offerContext)
                TravelOffer.update({"Bought Tag":random.choice([0,1])})
                # print(TravelOffer)

                #Unique ID

                dataframe_list.append(TravelOffer["Travel Offer ID"])
                dataframe_list.append(TravelOffer["User ID"])
                #static profile
                dataframe_list.append(TravelOffer["TimeStamp"])
                dataframe_list.append(TravelOffer['Date Of Birth'])
                dataframe_list.append(TravelOffer['city'])
                dataframe_list.append(TravelOffer['country'])
                #==========================================
                dataframe_list.append(TravelOffer["Loyalty Card"])
                dataframe_list.append(TravelOffer["Payment Card"])
                dataframe_list.append(TravelOffer["PRM Type"])
                dataframe_list.append(TravelOffer["Preferred means of transportation"])
                dataframe_list.append(TravelOffer["Preferred carrier"])
                dataframe_list.append(TravelOffer["Class"])
                dataframe_list.append(TravelOffer["Seat"])
                dataframe_list.append(TravelOffer["Refund Type"])
                #Category
                dataframe_list.append(TravelOffer['Quick'])
                dataframe_list.append(TravelOffer['Reliable'])
                dataframe_list.append(TravelOffer['Cheap'])
                dataframe_list.append(TravelOffer['Comfortable'])
                dataframe_list.append(TravelOffer['Door-to-door'])
                dataframe_list.append(TravelOffer['Environmentally friendly'])
                dataframe_list.append(TravelOffer['Short'])
                dataframe_list.append(TravelOffer['Multitasking'])
                dataframe_list.append(TravelOffer['Social'])
                dataframe_list.append(TravelOffer['Panoramic'])
                dataframe_list.append(TravelOffer['Healthy'])
                # dataframe_list.append(TravelOffer['Secure'])
                # dataframe_list.append(TravelOffer['Safe'])
                #legs_num
                dataframe_list.append(TravelOffer['Legs Number'])
                #profile
                dataframe_list.append(TravelOffer['Profile'])
                #Source-Des-Via
                dataframe_list.append(TravelOffer['Starting point'])
                dataframe_list.append(TravelOffer['Destination'])
                dataframe_list.append(TravelOffer['Via'])
                #legs_info
                dataframe_list.append(TravelOffer['LegMode'])
                dataframe_list.append(TravelOffer['LegCarrier'])
                dataframe_list.append(TravelOffer['LegSeat'])
                dataframe_list.append(TravelOffer['LegLength'])
                #context
                dataframe_list.append(TravelOffer["Departure time"])
                dataframe_list.append(TravelOffer["Arrival time"])
                dataframe_list.append(TravelOffer["Services"])
                dataframe_list.append(TravelOffer["Transfers"])
                dataframe_list.append(TravelOffer["Transfer duration"])
                dataframe_list.append(TravelOffer["Walking distance to stop"])
                dataframe_list.append(TravelOffer["Walking speed"])
                dataframe_list.append(TravelOffer["Cycling distance to stop"])
                dataframe_list.append(TravelOffer["Cycling speed"])
                dataframe_list.append(TravelOffer["Driving speed"])

                dataframe_list.append(TravelOffer['Bought Tag'])
                # print(dataframe_list)
                dataframe_lists.append(dataframe_list)
        df = pd.DataFrame(data=dataframe_lists)
        df.columns =["Travel Offer ID","User ID","TimeStamp",'Date Of Birth','city','country',
                     "Loyalty Card","Payment Card","PRM Type","Preferred means of transportation",
                     "Preferred carrier","Class","Seat","Refund Type",'Quick','Reliable','Cheap',
                     'Comfortable','Door-to-door','Environmentally friendly','Short','Multitasking',
                     'Social','Panoramic','Healthy','Legs Number','Profile','Starting point',
                     'Destination','Via','LegMode','LegCarrier','LegSeat','LegLength',"Departure time",
                     "Arrival time","Services","Transfers","Transfer duration","Walking distance to stop",
                     "Walking speed","Cycling distance to stop","Cycling speed","Driving speed","Bought Tag"] 
        
        return df

    #++++++++++RULE RANDOM++++++++++++++++
    def rule_random_index(self,proba):
        start = 0
        index = 0
        randnum = random.randint(1, sum(proba))
        for index, scope in enumerate(proba):
            start += scope
            if randnum <= start:
                break
        return index

    #+++++++++++++++++++++++++++++++
    def RULE_staticProfile_Generator(self):
        self.dataRange_StaticProfile()

        #personal Information-date of birth
        # arr = ['[10,18)', '[18,30)', '[30,40)','[40,60)','[60,80]'] 
        proba = [10,30,20,30,10]
        index = self.rule_random_index(proba)
        if index == 0:
            #age = random.randint(10,17)
            start_birth =(2003,1,1,0,0,0,10,10,10)#18
            end_birth = (2011,12,31,23,59,59,10,10,10)#10
            start = time.mktime(start_birth)
            end =time.mktime(end_birth)
            s =random.randint(start,end)

        elif index == 1:
            #age = random.randint(18,29)
            start_birth =(1991,1,1,0,0,0,10,10,10)#30
            end_birth = (2002,12,31,23,59,59,10,10,9)#18
            start = time.mktime(start_birth)
            end =time.mktime(end_birth)
            s =random.randint(start,end)

        elif index == 2: 
            #age = random.randint(30,39)
            start_birth =(1981,1,1,0,0,0,10,10,10)#40
            end_birth = (1990,12,31,23,59,59,10,10,9)#30
            start = time.mktime(start_birth)
            end =time.mktime(end_birth)
            s =random.randint(start,end)

        elif index == 3:
            #age = random.randint(40,59)
            start_birth =(1961,1,1,0,0,0,10,10,10)#60
            end_birth = (1980,12,31,23,59,59,10,10,9)#40
            start = time.mktime(start_birth)
            end =time.mktime(end_birth)
            s =random.randint(start,end)

        elif index == 4:
            #age = random.randint(60,80)
            start_birth =(1941,1,1,0,0,0,10,10,10)#80
            end_birth = (1960,12,31,23,59,59,10,10,9)#60
            start = time.mktime(start_birth)
            end =time.mktime(end_birth)
            s =random.randint(start,end)

        date_str = datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=s)
        date_of_birth = date_str.strftime("%Y-%m-%d")
        # print(date_of_birth)

        #-------------------------------------
        #country
        country = random.choice(self.countries)
        # print(country)

        #-------------------------------------
        #city
        city_search_name = "self.city_for_"+country
        city = random.choice(eval(city_search_name))

        static_profile_dict ={"Date Of Birth":date_of_birth,"city":city,"country":country}
        # print("\nRandom Registration Information has been generated:\n")
        # print(self.registration_dict)
        return static_profile_dict

    def RULE_dynamicProfile_Generator(self,staticProfile):
        self.dataRange_DynamicProfile()

        age = int(datetime.datetime.now().year) - int(staticProfile['Date Of Birth'][:4])
        country =staticProfile['country']
        
        #-------------------------------------
        #PRM
        PRM_type =[]
        #[60,] => Old
        if age >= 60 :
            PRM_type.append('Older person')

        #[22,35] => 10% Pregnant
        if age >= 22 and age <=35 :
            proba =[10,90]
            index =self.rule_random_index(proba)
            if index == 0:
                PRM_type.append('Pregnant woman')

        #[22,] => 10% carrycots
        if age >= 22 :
            proba = [10,90]
            index =self.rule_random_index(proba)
            if index == 0 :
                PRM_type.append("Persons porting a carrycots")

        #5% people need wheelchair, and can have only one type
        proba = [5,95]
        index =self.rule_random_index(proba)
        if index ==0 :
            wheelchairs =["Persons with impairments in their members / users of temporary wheelchair",
                         "Wheelchair users in mainstreaming seat",
                         "Wheelchair users in specific seat named h-seat"]
            wheelchair = random.choice(wheelchairs)
            PRM_type.append(wheelchair)

        #4% people blind or deafness or both
        proba = [4,96] 
        index =self.rule_random_index(proba)
        if index == 0:
            prm = ["Persons with blind or visual impairments",
                   "Person with deafness or auditory impairments"]
            num_of_PRM = random.randint(1,2)
            PRM_type += random.sample(prm,num_of_PRM)
        # print(PRM_type)

        #------------------------------------
        #payment cards
        payment_card = []

        #[,18] =>  No payment cards
        if age <=18 :
            payment_card = []
        else:
            #[18,40] => 20% apple/ google
            if age <=40 :
                proba = [20,95]
                index = self.rule_random_index(proba)
                if index ==0 :
                    num_of_paymentcards = random.randint(1,2)
                    cards =["Google Wallet","Apple Wallet"]
                    payment_card += random.sample(cards,num_of_paymentcards)

            
            #70% => mastercard / visa
            proba = [70,30]
            index = self.rule_random_index(proba)
            if index == 0:
                num_of_paymentcards = random.randint(1,2)
                cards =["Mastercard","Visa"]
                payment_card += random.sample(cards,num_of_paymentcards)    

            #40% =>paypal
            proba = [40,60]
            index = self.rule_random_index(proba)
            if index == 0:
                payment_card.append('Paypal')

        #---------------------------------------
        #Preferred Carrier
        #TODO 
        preferred_carrier = []
        for i in range(len(self.carriers)):
            preferred_carrier.append(random.randint(0,10)/2)
        # print(preferred_carrier)

        #--------------------------------------
        #class
        #Default Value
        class_type='Economy'

        #TODO purpose
        #prm => 70% first
        if not PRM_type == []:
            proba = [20,10,70]
        else:
            #age
            #[,25] => 70% Economy 25% business 5% first
            if age <=25:
                proba = [70,25,5]
            #[25,] => 60% Economy 30% business 10% first
            else:
                proba = [60,30,10]

        index =self.rule_random_index(proba)
        if index == 0 :
            class_type ="Economy"
        if index ==1 :
            class_type ="Business"
        if index ==2 :
            class_type = "First Class" 
        # print(class_type)


        #--------------------------------------
        #preferred transportation
        preferred_transportation =[]
        #TODO leg counts
        #TODO Starting point & des
        #TODO preferred carrier
        #class 
        #90% first/business man => air/ship/taxi
        if class_type == "First Class" or class_type == "Business":
            proba = [90,10]
            index =self.rule_random_index(proba)
            if index ==0:
                trans = ["Airline","Ship","Taxi"]
                num_of_transportation = random.randint(1,3)
                preferred_transportation +=random.sample(trans,num_of_transportation)
        #90% economy man => bus/metro/bike/car sharing/...
        elif class_type == 'Economy':
            proba = [90,10]
            index = self.rule_random_index(proba)
            if index == 0 :
                trans = ["Coach","Toll","Car Sharing","Train",
                       "Urban","Trolely Bus","Tram","Intercity","Metro","Bus","Other",
                       "Park","Bike Sharing"]
                    #    ["Coach","Toll","Car Sharing","Train",
                    #    "Urban","Trolely Bus","Tram","Intercity","Metro",
                    #    "Cable Way","Funicular","Bus","Other",
                    #    "Park","Bike Sharing"]
                num_of_transportation = random.randint(1,13)
                preferred_transportation +=random.sample(trans,num_of_transportation)  

        #prm
        prm_limit1 = ["Older person",
                "Persons with impairments in their members / users of temporary wheelchair",
                "Persons porting a carrycots",
                "Wheelchair users in mainstreaming seat",
                "Wheelchair users in specific seat named h-seat",
                "Pregnant woman"] 

        prm_limit2 =[
                "Persons with blind or visual impairments",
                "Persons porting a carrycots",
                "Person with deafness or auditory impairments"]

        #30% normal people => cable/funicular
        if PRM_type == []:
            proba = [30,70]
            index =self.rule_random_index(proba)
            if index ==0:
                trans = ["Cable Way","Funicular"]
                num_of_transportation = random.randint(1,2)
                preferred_transportation +=random.sample(trans,num_of_transportation)  
        else:
            for n in range(len(PRM_type)):
                #wheelchair/pregnant/old => NO cable
                if PRM_type[n] in prm_limit1:
                    if 'Cable Way' in preferred_transportation:
                        preferred_transportation.remove('Cable Way') 
                    if 'Funicular' in preferred_transportation:
                        preferred_transportation.remove('Funicular')

                #blind/carrycots/deafness => NO bike
                elif PRM_type[n] in prm_limit2:
                    if 'Bike Sharing' in preferred_transportation:
                        preferred_transportation.remove('Bike Sharing')
        # print(preferred_transportation)

        #-----------------------------------------------
        #loyalty card
        # ["Cartafreccia","FlyingBlue","Golden Card","Grand Voyageur"]
        loyalty_card=[]

        #preferred carrier
        # carriers =["Trenitalia","SNFC","AirFrance","VBB","TMB","Renfe","RegioJet","KLM","Iberia","FlixBus"]
        # Trenitalia,Renfe >=4 => 70% cartafreccia
        if preferred_carrier[0] >=4 or preferred_carrier[5]>=4 :
            proba=[70,30]
            index=self.rule_random_index(proba)
            if index ==0:
                loyalty_card.append('Cartafreccia')
        # AirFrance, RegioJet,KLM,Iberia => 70% FlyingBlue
        if preferred_carrier[2] >=4 or preferred_carrier[6]>=4 or preferred_carrier[7]>=3.5 or preferred_carrier[8]>=4:
            proba=[70,30]
            index=self.rule_random_index(proba)
            if index ==0:
                loyalty_card.append('FlyingBlue') 

        #SNFC =>70% Grand Voyageur
        if preferred_carrier[1]>=3.5:
            proba=[70,30]
            index=self.rule_random_index(proba)
            if index ==0:
                loyalty_card.append('Grand Voyageur')    

        #preferred transportation
        #30% airline => flyingBlue / Grand Voyageur
        if 'Airline' in preferred_transportation:
            proba=[30,70]
            index =self.rule_random_index(proba)
            if index == 0:
                num_of_cards = random.randint(1,2)
                loyalty_card += random.sample(['FlyingBlue','Grand Voyageur'],num_of_cards) 
        #30% train => Cartafreccia
        if 'Train' in preferred_transportation:
            proba=[30,70]
            index =self.rule_random_index(proba)
            if index == 0:
                loyalty_card.append('Cartafreccia')

        #country
        # 50% Italy => Cartafreccia
        if country =='Italy':
            proba=[50,50]
            index =self.rule_random_index(proba)
            if index == 0:
                loyalty_card.append('Cartafreccia')     
        # 50% France => Grand Voyageur
        if country == 'France':
            proba=[50,50]
            index =self.rule_random_index(proba)
            if index == 0:
                loyalty_card.append('Grand Voyageur')      
                
        
        # 10% people  => Golden Card
        proba=[10,90]
        index =self.rule_random_index(proba)
        if index == 0:
            loyalty_card.append('Golden Card')

        #Remove Duplication
        loyalty_card = list(set(loyalty_card))
        # print(loyalty_card)

        #---------------------------------------------------
        #seat
        #Default Value
        seat_type = random.choice(self.seats)

        #age
        #70% young => Window
        if age <=30 :
            proba =[70,30]
            index =self.rule_random_index(proba)
            if index ==0:
                seat_type = 'Window' 

        #preferred transportation
        #60% ship => window
        if 'Ship' in preferred_transportation:
            proba =[60,40]
            index =self.rule_random_index(proba)
            if index ==0:
                seat_type = 'Window'  

        #TODO Arrival time -Departure time

        #PRM
        #disabled people => 85%large/ 10% aisle /5% window
        if not PRM_type == []:
            proba =[85,10,5]
            index =self.rule_random_index(proba)
            if index ==0:
                seat_type = 'Large'
            if index == 1:
                seat_type = 'Aisle'
            if index== 2:
                seat_type ='Window'
        # print(seat_type)

        #---------------------------------------------------
        #refund
        #default value-70% people => automatic way
        proba =[70,30]
        index =self.rule_random_index(proba)
        if index == 0:
            refund_type = "Automatic refund"
        if index == 1:
            refund_type = "Manual refund"

        #age
        #90% old people => manual refund
        if age >=60 :
            proba =[90,10]
            index=self.rule_random_index(proba)
            if index ==0:
                refund_type ="Manual refund"
        
        #prm
        #blind => manual
        if "Persons with blind or visual impairments" in PRM_type:
            refund_type = "Manual refund"
        #deafness => automatic
        if "Person with deafness or auditory impairments" in PRM_type:
            refund_type= "Automatic refund"
        # print(refund_type)

        #-----------------------------------------------------
        #all
        dynamic_profile_dict = {"Loyalty Card":str(loyalty_card),
                           "Payment Card":str(payment_card),
                           "PRM Type":str(PRM_type),
                           "Preferred means of transportation":str(preferred_transportation),
                           "Preferred carrier":str(preferred_carrier),
                           "Class":class_type,
                           "Seat":seat_type,
                           "Refund Type":refund_type}

        # print("\nRandom Preference Information has been generated:\n")
        # print(self.preference_dict)
        return dynamic_profile_dict
  
    def RULE_offerContext_Generator(self,staticProfile,dynamicProfile):
        self.dataRange_OfferContext()

        age = int(datetime.datetime.now().year) - int(staticProfile['Date Of Birth'][:4])
        country =staticProfile['country']
        city =staticProfile['city']
        PRM_type =dynamicProfile['PRM Type']
        class_type =dynamicProfile['Class']
        preferred_transportation = dynamicProfile['Preferred means of transportation']
        #--------------------------------------------------
        #purpose
        #default value
        profile_type =random.choice(self.profile_typies)

        #age
        #children & old people => NO Business
        if age <=18 or age>=60 :
            profile_type = random.choice(["Basic","Family","Leisure"])
        # print(profile_type)

        #-------------------------------------------------
        #starting point
        #Default Value
        start_point = random.choice(self.starting_points)

        #city
        #40% people => same starting point with city
        proba =[40,60]
        index=self.rule_random_index(proba)
        if index == 0:
            start_point = city
        # print(start_point)

        #---------------------------------------------------
        #destination
        #Default value
        destination = random.choice(self.destination_points)

        #10% people => same des with starting point
        proba =[10,90]
        index=self.rule_random_index(proba)
        if index == 0:
            destination = start_point 

        #TODO prm
        # print(destination)

        #---------------------------------------------------
        #Via Stops (des > prm > purpose >age)
        via_stop=[]
        via_dict = ["Brugge","Bruxelle","Praha","Krumlov","Paris","Nice",
                       "Frankfurt","Berlin","Dublin","Crok","Milan","Rome",
                       "Warsaw","Kraków","Lisbon","Porto","Barcelona","Madrid",
                       "London","Oxford"]
        via_dict.remove(start_point)
        num_of_via = random.randint(0,4)

        #des
        #same source and des => NO leg
        if destination == start_point:
            num_of_via = 0


        #prm
        #95% disabled people =>[0,1] leg 
        elif not PRM_type == []:
            proba=[95,5]
            index=self.rule_random_index(proba)
            if index == 0:
                num_of_via= random.randint(0,1)

        #purpose
        #90% business => no more than 2 legs
        elif profile_type == 'Business':
            proba =[90,10]
            index=self.rule_random_index(proba)
            if index ==0 :
                num_of_via = random.randint(0,2)
        
        #age
        # 70% young => [3,5] legs
        elif age <=35 and age>= 18:
            proba =[70,30]
            index=self.rule_random_index(proba)
            if index ==0:
                num_of_via =random.randint(3,5)

        if destination  != start_point:
            via_dict.remove(destination) 
        via_stop = random.sample(via_dict,num_of_via)
        legs_num = num_of_via + 1

        #------------------------------------------------
        #departure_time
        #TODO his records have only 1/2 offers at same time 
        
        #for new offer
        # start_departure = time.localtime() 
        #for his
        #start time later than birthday
        birthDay =staticProfile['Date Of Birth']
        start_departure = time.strptime(birthDay,'%Y-%m-%d')
        # start_departure = (1990,1,1,0,0,0,0,0,0)
        end_departure =(2022,12,31,0,0,0,0,0,0) 
        start_dep = time.mktime(start_departure)
        end_dep =time.mktime(end_departure)
        sec_dep =random.randint(start_dep,end_dep)
        date_dep = datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=sec_dep)
        departure_time = date_dep.strftime("%Y-%m-%d")

        # class
        #first/business => 90% more siutable timetable
        if class_type == 'First Class' or class_type== 'Business':
            proba = [90,10]
            index =self.rule_random_index(proba)
            if index ==0:
                start_hour = random.randint(10,16)
            if index ==1:
                start_hour = random.randint(5,23)
        else:
            start_hour =random.randint(5,23)
        
        departure_time = departure_time + ' ' + str(start_hour) +':'+str(random.randint(0,59))
        # print(departure_time)

        #------------------------------------------------
        #arrival time
        timeArray = time.strptime(departure_time,"%Y-%m-%d %H:%M")
        start_arr= int(time.mktime(timeArray))
        #Default Value
        end_arr = start_arr + 3600*2

        #des
        #same city => 90% less than 2 hours 10% no more than 5 hours
        if destination == start_point:
            proba=[90,10]
            index = self.rule_random_index(proba)
            if index ==0:
                time_interval = 3600 * 2
                end_arr = start_arr + time_interval
            if index==1:
                time_interval =3600 * 5
                end_arr = start_arr + time_interval

        else:
            #via stops
            via_count = len(via_stop)
            # leg=0 => 90% no more than 4 hours 10% no more than 1 day
            if via_count == 0:
                proba = [90,10]
                index = self.rule_random_index(proba)
                if index==0:
                    time_interval =3600*4
                    end_arr =start_arr + time_interval
                if index ==1:
                    time_interval =3600*24
                    end_arr =start_arr+time_interval
                    start_arr = start_arr + 3600*4
            
            #legs [1,3] => 60% no more than 2 days(at least 10 hours) 
            #              40% no more than 4 days(at least 2 days) 
            elif via_count <= 3:
                proba = [60,40]
                index = self.rule_random_index(proba)
                if index==0:
                    time_interval =3600*24*2
                    end_arr =start_arr + time_interval
                    start_arr =start_arr+ 3600*10
                if index ==1:
                    time_interval =3600*24*4
                    end_arr =start_arr+time_interval
                    start_arr= start_arr+ 3600*24*2
            
            #legs [3,] =>  80% no more than 5 days(at least 2 days) 
            #              20% no more than 7 days(at least 5 days) 
            elif via_count > 3 :
                proba=[80,20]
                index = self.rule_random_index(proba)
                if index ==0:
                    time_interval =3600*24*5
                    end_arr = start_arr +time_interval
                    start_arr =start_arr+3600*24*2
                if index ==1:
                    time_interval =3600*24*7
                    end_arr=start_arr+time_interval
                    start_arr=start_arr+3600*24*5

        sec_arr = random.randint(start_arr,end_arr)
        date_arr = datetime.datetime(1970,1,1) + datetime.timedelta(seconds=sec_arr)
        arrival_time = date_arr.strftime("%Y-%m-%d %H:%M") 
        # print(arrival_time)
        
        #------------------------------------------------------
        #services
        service =[]

        #preferred transportation
        #Airline => 70% Air
        if 'Airline' in preferred_transportation:
            proba=[70,30]
            index =self.rule_random_index(proba)
            if index == 0:
                service.append('Air')
        # 
        if 'Train' in preferred_transportation:
            proba=[70,30]
            index =self.rule_random_index(proba)
            if index == 0:
                service.append('Highspeed Train')
        # Coach => 70% coach
        if 'Coach' in preferred_transportation:
            proba=[70,30]
            index =self.rule_random_index(proba)
            if index == 0:
                service.append('Coach')
        #Urban =>70% suburban
        if 'Urban' in preferred_transportation:
            proba=[70,30]
            index =self.rule_random_index(proba)
            if index == 0:
                service.append('Suburban')
        #tram => 70% tram
        if 'Tram' in preferred_transportation:
            proba=[70,30]
            index =self.rule_random_index(proba)
            if index == 0:
                service.append('Tram')
        #Intercity => 70% local rail
        if 'Intercity' in preferred_transportation:
            proba=[70,30]
            index =self.rule_random_index(proba)
            if index == 0:
                service.append('Local rail')
        #Metro => 70% underground
        if 'Metro' in preferred_transportation:
            proba=[70,30]
            index =self.rule_random_index(proba)
            if index == 0:
                service.append('Underground')
        #ship => 70% water transport
        if 'Ship' in preferred_transportation:
            proba=[70,30]
            index =self.rule_random_index(proba)
            if index == 0:
                service.append('Water transport')
        #cable => 70% Telecabin
        if 'Cable Way' in preferred_transportation:
            proba=[70,30]
            index =self.rule_random_index(proba)
            if index == 0:
                service.append('Telecabin')
        #Funicular => 70% local type
        if 'Funicular' in preferred_transportation:
            proba=[70,30]
            index =self.rule_random_index(proba)
            if index == 0:
                service.append('Local type')
        #Bus => 70% Bus / DRT
        if 'Bus' in preferred_transportation:
            proba=[70,30]
            index =self.rule_random_index(proba)
            if index == 0:
                num_of_services =random.randint(1,2)
                service +=random.sample(['Bus','DRT'],num_of_services)

        #10% people choose midcellaneous
        proba =[10,90]
        index =self.rule_random_index(proba)
        if index ==0:
            service.append('Miscellaneous')

        if service == []:
            service.append(random.choice(self.services))
        # print(service)

        #----------------------------------------------------
        #transfers
        # ["Unlimited","None","Max 1","Max 2","Max 3","Max 4"]
        transfer = random.choice(self.transfers)

        #prm
        #Disabled people => 40% None 30% Max_1 20% Max_2 10% Other
        if not PRM_type == []:
            proba = [40,30,20,10]
            index =self.rule_random_index(proba)
            if index ==0:
                transfer = 'None'
            if index ==1:
                transfer = "Max 1"
            if index ==2 :
                transfer = 'Max 2'
            if index ==3:
                transfer = random.choice(['Unlimited','Max 3','Max 4'])

        # print(transfer)

        #------------------------------------------------------
        #walking distance (prm > purpose > age)
        
        #prm
        #Disabled people => 80% short distance
        if not PRM_type ==[]:
            proba = [80,20]
 
        #purpose
        #Business => 90% short
        elif profile_type =='Business':
            proba = [90,10]

        #age
        # old people => 60% short
        elif age >= 55 :
            proba =[60,40]
        
        #Default Value
        else:
            proba =[50,50]

        index =self.rule_random_index(proba)
        if index ==0:
            walking_distance_to_stop = str(random.randrange(500,2000,500))+'m'
        if index ==1:
            walking_distance_to_stop = str(random.randrange(2000,5000,500))+'m'

        # print(walking_distance_to_stop)

        #------------------------------------------------------
        #transfer duration
        # ["normal","At least 10 min","At least 15 min","At least 20 min",
        #"At least 25 min","At least 30 min","At least 35 min",
        #"At least 40 min","At least 45 min"]
        #Default Value
        transfer_duration = random.choice(self.transfer_durations)

        #prm
        #disabled people => more time
        if not PRM_type ==[]:
            proba = [40,30,20,10]
 
        # purpose
        #Business => less time
        elif profile_type == 'Business':
            proba = [5,10,20,65] 

        #transfers
        #more transfers => less time
        elif transfer in ['Unlimited','Max 3','Max 4']:
            proba = [10,20,30,40]

        #walking distance
        #far => more time
        elif int(walking_distance_to_stop[:-1]) >= 2500 :
            proba =[40,40,15,5]

        else:
            proba = [25,25,25,25]


        index =self.rule_random_index(proba)
        if index ==0:
            transfer_duration = random.choice(["At least 40 min","At least 45 min"])
        if index ==1:
            transfer_duration = random.choice(["At least 30 min","At least 35 min"])
        if index ==2 :
            transfer_duration = random.choice(["At least 20 min","At least 25 min"])
        if index ==3:
            transfer_duration = random.choice(["normal","At least 10 min","At least 15 min"])             

        # print(transfer_duration)

        #-----------------------------------------------------
        #walking speed(prm > purpose >age)
        # ["Slow","Medium","Fast"]
        walking_speed = random.choice(self.walking_speads)

        #prm
        #Disabled people => 80% slow 18% medium 2%fast
        if not PRM_type ==[]:
            proba = [80,18,2]
 
        #purpose
        #Business => 5% slow 15% medium 80% fast
        elif profile_type =='Business':
            proba = [5,15,80]

        #age
        # old people => 60% slow 20% medium 20% fast
        elif age >= 55 :
            proba =[60,20,20]
        
        #walking distance
        #far => 60% fast
        elif int(walking_distance_to_stop[:-1]) >= 2500:
            proba = [10,30,60]
        #normal dis. =>60% medium
        elif int(walking_distance_to_stop[:-1]) >= 1000:
            proba = [10,60,30]
        #close =>60% slow
        elif int(walking_distance_to_stop[:-1]) <= 1000:
            proba = [60,30,10] 
        
        #Default Value
        else:
            proba =[33,34,33]

        index =self.rule_random_index(proba)
        if index ==0:
            walking_speed = 'Slow'
        if index ==1:
            walking_speed = 'Medium'
        if index ==2:
            walking_speed = 'Fast'
        # print(walking_speed)

        #---------------------------------------------------
        #cycling distance
        cycling_distance_to_stop = str(random.randrange(2000,20000,500))+'m'
        
        #prm
        #Disabled people => NO cycling
        if not PRM_type ==[]:
            cycling_distance_to_stop = '0m'

        else:
            #purpose
            #Business => 90% short
            if profile_type =='Business':
                proba = [90,10]

            #age
            # old people => 60% short
            elif age >= 55 :
                proba =[60,40]

            #Default Value
            else:
                proba =[50,50]

            index =self.rule_random_index(proba)
            if index ==0:
                cycling_distance_to_stop = str(random.randrange(2000,10000,500))+'m'
            if index ==1:
                cycling_distance_to_stop = str(random.randrange(10000,20000,500))+'m'
        # print(cycling_distance_to_stop)

        #----------------------------------------------------
        #cycling speed
        cycling_speed = random.choice(self.cycling_speads)

        #prm
        #Disabled people => NO cycling
        if not PRM_type ==[]:
            cycling_speed = '0'

        else:
            #purpose
            #Business => 5% slow 15% medium 80% fast
            if profile_type =='Business':
                proba = [5,15,80]

            #age
            # old people => 60% slow 20% medium 20% fast
            elif age >= 55 :
                proba =[60,20,20]
        
            #cycling distance
            #far => 60% fast
            elif int(cycling_distance_to_stop[:-1]) >= 15000:
                proba = [10,30,60]
            #normal dis. =>60% medium
            elif int(cycling_distance_to_stop[:-1]) >= 5000:
                proba = [10,60,30]
            #close =>60% slow
            elif int(cycling_distance_to_stop[:-1]) <= 5000:
                proba = [60,30,10] 
        
            #Default Value
            else:
                proba =[33,34,33]

            index =self.rule_random_index(proba)
            if index ==0:
                cycling_speed = 'Slow'
            if index ==1:
                cycling_speed = 'Medium'
            if index ==2:
                cycling_speed = 'Fast'

        # print(cycling_speed)

        #-------------------------------------------------
        #driving speed
        driving_speed = random.choice(self.driving_speeds)

        #prm
        #disabled people => NO drive
        if not PRM_type ==[]:
            driving_speed='0'

        #age
        #young and old => No drive
        elif age <= 18 or age >=60 :
            driving_speed ='0'

        else:
            
            #purpose
            #Business => 5% slow 15% medium 80% fast
            if profile_type =='Business':
                proba = [5,15,80]

            #age
            # [18,30] => 20% slow 20% medium 60% fast
            elif age <=30 :
                proba =[20,20,60] 
            # [30,60] => 50% slow 30% medium 20% fast
            elif age <60 :
                proba = [50,30,20]

            #default value
            else:
                proba =[33,34,33]

            index =self.rule_random_index(proba)
            if index ==0:
                cycling_speed = 'Slow'
            if index ==1:
                cycling_speed = 'Medium'
            if index ==2:
                cycling_speed = 'Fast' 

        # print(driving_speed)

        #---------------------------------------------------------------
        # print("\nRandom Trip Planner Information has been generated:\n")
        OfferContext_dict = {"Legs Number":legs_num,"Profile":profile_type,
                   "Starting point":start_point,"Destination":destination,
                   "Departure time":departure_time,"Arrival time":arrival_time,
                   "Via":str(via_stop),"Services":str(service),"Transfers":transfer,
                   "Transfer duration":transfer_duration,
                   "Walking distance to stop":walking_distance_to_stop,
                   "Walking speed":walking_speed,
                   "Cycling distance to stop":cycling_distance_to_stop,
                   "Cycling speed":cycling_speed,
                   "Driving speed":driving_speed}

        # print(TP_dict)
        return OfferContext_dict

    def RULE_historical_Generator(self,username,staticChangeTag=True,staProfile=None):

        dataframe_lists = []
        unique_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S')      
        
        userID = username
        if staticChangeTag is True:
            staticProfile = self.RULE_staticProfile_Generator()
        else:
            staticProfile = staProfile
            if staProfile is None:return

        TimeStamp = datetime.datetime.now()

        TravelOffer = {"TimeStamp":TimeStamp}
        TravelOffer.update({"User ID":userID })
        TravelOffer.update(staticProfile)
            
        #change the number to generate different amount of his records
        dynamic_num =random.randint(4,5)
        for inx_d in range(dynamic_num):
            dynamicProfile = self.RULE_dynamicProfile_Generator(staticProfile)
            TravelOffer.update(dynamicProfile)

            travel_num =random.randint(20,25)
            for n in range(travel_num):
                dataframe_list = []
                travelOfferID = str(unique_id) + str(n)
                # print(legs_num)

                category = self.offer_Categorizer()
                offerContext = self.RULE_offerContext_Generator(staticProfile,dynamicProfile)
                legs_num = offerContext['Legs Number']
                legInfo = self.leg_info_Generator(legs_num)

                TravelOffer.update({"Travel Offer ID":travelOfferID})
                TravelOffer.update(category)
                TravelOffer.update(legInfo)
                TravelOffer.update(offerContext)
                TravelOffer.update({"Bought Tag":random.choice([0,1])})
                # print(TravelOffer)

                #Unique ID

                dataframe_list.append(TravelOffer["Travel Offer ID"])
                dataframe_list.append(TravelOffer["User ID"])
                #static profile
                dataframe_list.append(TravelOffer["TimeStamp"])
                dataframe_list.append(TravelOffer['Date Of Birth'])
                dataframe_list.append(TravelOffer['city'])
                dataframe_list.append(TravelOffer['country'])
                #==========================================
                dataframe_list.append(TravelOffer["Loyalty Card"])
                dataframe_list.append(TravelOffer["Payment Card"])
                dataframe_list.append(TravelOffer["PRM Type"])
                dataframe_list.append(TravelOffer["Preferred means of transportation"])
                dataframe_list.append(TravelOffer["Preferred carrier"])
                dataframe_list.append(TravelOffer["Class"])
                dataframe_list.append(TravelOffer["Seat"])
                dataframe_list.append(TravelOffer["Refund Type"])
                #Category
                dataframe_list.append(TravelOffer['Quick'])
                dataframe_list.append(TravelOffer['Reliable'])
                dataframe_list.append(TravelOffer['Cheap'])
                dataframe_list.append(TravelOffer['Comfortable'])
                dataframe_list.append(TravelOffer['Door-to-door'])
                dataframe_list.append(TravelOffer['Environmentally friendly'])
                dataframe_list.append(TravelOffer['Short'])
                dataframe_list.append(TravelOffer['Multitasking'])
                dataframe_list.append(TravelOffer['Social'])
                dataframe_list.append(TravelOffer['Panoramic'])
                dataframe_list.append(TravelOffer['Healthy'])
                # dataframe_list.append(TravelOffer['Secure'])
                # dataframe_list.append(TravelOffer['Safe'])
                #legs_num
                dataframe_list.append(TravelOffer['Legs Number'])
                #profile
                dataframe_list.append(TravelOffer['Profile'])
                #Source-Des-Via
                dataframe_list.append(TravelOffer['Starting point'])
                dataframe_list.append(TravelOffer['Destination'])
                dataframe_list.append(TravelOffer['Via'])
                #legs_info
                dataframe_list.append(TravelOffer['LegMode'])
                dataframe_list.append(TravelOffer['LegCarrier'])
                dataframe_list.append(TravelOffer['LegSeat'])
                dataframe_list.append(TravelOffer['LegLength'])
                #context
                dataframe_list.append(TravelOffer["Departure time"])
                dataframe_list.append(TravelOffer["Arrival time"])
                dataframe_list.append(TravelOffer["Services"])
                dataframe_list.append(TravelOffer["Transfers"])
                dataframe_list.append(TravelOffer["Transfer duration"])
                dataframe_list.append(TravelOffer["Walking distance to stop"])
                dataframe_list.append(TravelOffer["Walking speed"])
                dataframe_list.append(TravelOffer["Cycling distance to stop"])
                dataframe_list.append(TravelOffer["Cycling speed"])
                dataframe_list.append(TravelOffer["Driving speed"])

                dataframe_list.append(TravelOffer['Bought Tag'])
                # print(dataframe_list)
                dataframe_lists.append(dataframe_list)
        df = pd.DataFrame(data=dataframe_lists)
        df.columns =["Travel Offer ID","User ID","TimeStamp",'Date Of Birth','city','country',
                     "Loyalty Card","Payment Card","PRM Type","Preferred means of transportation",
                     "Preferred carrier","Class","Seat","Refund Type",'Quick','Reliable','Cheap',
                     'Comfortable','Door-to-door','Environmentally friendly','Short','Multitasking',
                     'Social','Panoramic','Healthy','Legs Number','Profile','Starting point',
                     'Destination','Via','LegMode','LegCarrier','LegSeat','LegLength',"Departure time",
                     "Arrival time","Services","Transfers","Transfer duration","Walking distance to stop",
                     "Walking speed","Cycling distance to stop","Cycling speed","Driving speed","Bought Tag"] 
        
        return df 


    def res2csv(self,df,file_path,filename):
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        filename = file_path +'/'+filename 
        df.to_csv(filename,encoding='utf-8',mode='w',header=True,index=False)

    def _GenerateHis2File(self,username,staticChangeTag=True,staProfile=None):
        # df = self.historical_Generator(username,staticChangeTag=staticChangeTag,staProfile=staProfile)
        df = self.RULE_historical_Generator(username,staticChangeTag=staticChangeTag,staProfile=staProfile)
        self.res2csv(df, self.TEST_HISTORICAL_DATA_PATH, username+'.csv')
        print('\n>>>\nUser {} `s historical records has been generated in the file \n{}\n'.format(username,self.TEST_HISTORICAL_DATA_PATH))

class DATA_Population:

    his = TEST_HistoricalDataGenerate()
    #########################################################################################################
    ##########                                                                                     ##########
    ##########                                  RAW DATA===> ONE HOT DATA                          ##########
    ##########                                                                                     ##########
    #########################################################################################################

    def data_OneHot (self,df):
        #One_hot
        OneHotEncoder = OfferOneHotEncoder()
        df_onehot = OneHotEncoder.df_Regenerator(df)

        # timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        # self.his.res2csv(df_onehot,his.CHECK_USER2TSP_PATH,'checkOneHot_{}.csv'.format(timestamp))

        return df_onehot

    #########################################################################################################
    ##########                                                                                     ##########
    ##########                          DATA ====> Pure DATA (delete the same columns)             ##########
    ##########                                                                                     ##########
    #########################################################################################################

    def data_Pure (self,df):
        #Pure
        df_pure = df.loc[:, (df != df.iloc[0]).any()]

        # timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') 
        # self.his.res2csv(df_pure,self.CHECK_USER2TSP_PATH,'checkPure_{}.csv'.format(timestamp))

        return df_pure
    
    #########################################################################################################
    ##########                                                                                     ##########
    ##########                                   RAW DATA ===> X_RAW , Y_RAW                       ##########
    ##########                                                                                     ##########
    #########################################################################################################
    
    def label_data_split(self,df_raw,normalization=None):

        X_Raw= df_raw.drop(columns='Bought Tag')
        df_normalization = X_Raw.copy(deep=True)
        X_Raw = self.data_normalization(df_normalization,X_Raw,normalization)

        y_Raw = df_raw['Bought Tag']
        y_Raw = np.array(y_Raw)

        return X_Raw,y_Raw,df_normalization

    #########################################################################################################
    ##########                                                                                     ##########
    ##########                                     Normalization Method                            ##########
    ##########                                                                                     ##########
    #########################################################################################################
    def zscore_normalization(self,data_fit,data_calculate):
        scaler = preprocessing.StandardScaler().fit(data_fit) 
        data_scale = scaler.transform(data_calculate)
        # print(data_scale)
        return data_scale
    
    def min_max_normalization(self,data_fit,data_calculate):
        min_max_scaler = preprocessing.MinMaxScaler()
        minmax = min_max_scaler.fit(data_fit)
        data_minmax = minmax.transform(data_calculate)
        # df =pd.DataFrame(data=data_minmax)
        # df.to_csv('./Meeting1/minMaxNor.csv',mode='w',encoding='utf-8',header=True,index=False)
        return data_minmax

    def data_normalization(self,df_fit,df_cal,normalization):
        if normalization =='zero':
            narray_nor = self.zscore_normalization(df_fit,df_cal)
        elif normalization == 'minmax':
            narray_nor = self.min_max_normalization(df_fit,df_cal)
        else:
            narray_nor = np.array(df_cal)

        return narray_nor

    #########################################################################################################
    ##########                                                                                     ##########
    ##########                         X_RAW,y_RAW ===> TRAIN / TEST data                          ##########
    ##########                                                                                     ##########
    #########################################################################################################

    def train_test_split(self,X_Raw, Y_RAW, test_radio=0.2, seed=None):
        
        if seed:
            np.random.seed(seed)

        shuffled_indexes = np.random.permutation(len(X_Raw))
        test_size = int(len(X_Raw)*test_radio)

        test_indexes = shuffled_indexes[:test_size]
        train_indexes = shuffled_indexes[test_size:]

        X_train = X_Raw[train_indexes]
        Y_train = Y_RAW[train_indexes]

        X_test = X_Raw[test_indexes]
        Y_test = Y_RAW[test_indexes]

        return X_train, Y_train, X_test, Y_test

    def test_data_modified_on_old_model(self,df_test,feature_used):
        df_test_modified = pd.DataFrame()
        df_test_not_used = pd.DataFrame()
        for col in feature_used:
            if col in df_test.columns:
                df_test_modified[col] = df_test[col]
            else:
                df_test_modified[col] = 0
                
        columns = df_test.columns.tolist()
        for clm in columns:
            if clm not in feature_used:
                if df_test[clm] is not 0 :
                    
                    df_test_not_used[clm] = df_test[clm] 

        return df_test_modified,df_test_not_used
        
class CLASSIFIER_LEANER:

    his = TEST_HistoricalDataGenerate()
    ppl = DATA_Population()
    PT =ParametersTunning()


    recommender_categories = [
        'KNeighborsClassifier','SVC','DecisionTreeClassifier','LogisticRegression','RandomForestClassifier'
    ]


    def fetch_data_from_file(self,path,filename):
        _file = path +'/' + filename+'.csv'
        df_raw = pd.read_csv(_file)
        return df_raw
    
    def get_TrainData(self,username,seed,normalization='zero'):
        df_raw = self.fetch_data_from_file(self.his.TEST_HISTORICAL_DATA_PATH, username)
        del(df_raw['TimeStamp'])
        del(df_raw['Travel Offer ID'])
        del(df_raw['User ID'])
        df_onehot = self.ppl.data_OneHot(df_raw)

        if len(df_onehot) > 1 :
            df_pure =self.ppl.data_Pure(df_onehot)
        else:
            df_pure = df_onehot

        df_features =df_pure.drop(columns='Bought Tag')
        feature_used = df_features.columns.tolist()

        X_all,y_all,df_normalization = self.ppl.label_data_split(df_pure,normalization=normalization)
        X_t,y_t,X_ts,y_ts = self.ppl.train_test_split(X_all,y_all,seed=seed)
        TrainData_dict = {'X_train':X_t,'y_train':y_t,'X_test':X_ts,'y_test':y_ts,'X_all':X_all,'y_all':y_all,'feature_used':feature_used,'df_normalization':df_normalization}

        return TrainData_dict

    #########################################################################################################
    ##########                                                                                     ##########
    ##########       MODEL TRAINED BY ALL THE RS IN RS_Category and GIVE THE BEST ONE              ##########
    ##########                                                                                     ##########
    #########################################################################################################

    def model_training(self,username,recommender_input_data,model_version='latest',save_tag=True,fit_data_tag='all',recommender_category=None,file_path=None):
        save_file_name = 'best_model_{}_{}_{}'.format(username,fit_data_tag,model_version)    

        if recommender_category is None :
            recommender_category = self.recommender_categories
            
        #TODO split dataset based on the label 
        if fit_data_tag == 'train':
            train_data = recommender_input_data['X_train']
            train_label = recommender_input_data['y_train']
            evaluation_data =recommender_input_data['X_test']
            evaluation_label =recommender_input_data['y_test']
            best_model_info =  self.PT.get_best_recommender(recommender_category,train_data,train_label,evaluation_data,evaluation_label,save_tag=save_tag,save_file_name=save_file_name,file_path=file_path)

        if fit_data_tag == 'all':
            train_data = recommender_input_data['X_all']
            train_label = recommender_input_data['y_all']
            best_model_info = self.PT.get_best_recommender(recommender_category,train_data,train_label,save_tag=save_tag,save_file_name=save_file_name,file_path=file_path)

        best_model_info.update(recommender_input_data)
        self.PT.save_info(best_model_info,save_file_name,file_path=file_path)
        return best_model_info

    #########################################################################################################
    ##########                                                                                     ##########
    ##########                            API: USED TO TRAIN A MODEL                               ##########
    ##########                                                                                     ##########
    #########################################################################################################
    ## INPUT  :                                                                                            ##
    ##         -username: used to fetch the historical data of the user,same as the table name in TO DB    ##
    ##         -seed: used to set seed to split func & seed is set same as model_version for better use    ##
    ##         -fit_data_tag:['all','train'] 'all' means fit with all historical data                      ##
    ##                                       'train' means fit with train data and evaluate with test data ##
    ##         -recommender_category: Set Algorithm sets that will be used to find the bset model          ##
    ##                                None means use the default set defined in APIs_MODELTraining.py      ##
    ##=====================================================================================================##
    ## OUTPUT  :                                                                                           ##
    ##          -MODEL/best_model_username_fitTag_modelVersion.m as best model                             ##
    ##          -INFO/best_model_username_fitTag_modelVersion.npy as best model info                       ##
    ##          -Given the detialed info of every Algorithm in screen                                      ##
    #########################################################################################################

    def API_CLASSIFIER_TRAIN(self,username,seed=999,fit_data_tag='all',recommender_category=None,file_path=None):
        data = self.get_TrainData(username,seed)
        self.model_training(username,data,recommender_category=recommender_category,fit_data_tag=fit_data_tag,file_path=file_path)
    #########################################################################################################
    ##########                                                                                     ##########
    ##########                 EVALUATION ONE MODEL CREDIBILITY ON EVALUATION DATASET              ##########
    ##########                                                                                     ##########
    #########################################################################################################

    def model_evaluation(self,model_name,evaluation_data,evaluation_label,other_model_path=None):
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
        model = self.PT.load_model(model_name,file_path=other_model_path)
        best_info = self.PT.load_info(model_name,file_path=other_model_path)

        evaluation_score = self.PT.evaluation_on_model(model,evaluation_data,evaluation_label)
        model_score = best_info['best_score']

        model_credibility = abs(model_score - evaluation_score)/evaluation_score
        print('>>> Model score is {}\n'.format(model_score))
        print(">>> After evaluation with the given evaluation set, the model credibility is ",model_credibility)
        print('-----------------------------------------------------------------------------------------------\n')
        return model_credibility

    def CHECK_ADMINISTRATOR2CLASSIFIER_TRAIN(self,username):

        self.his._GenerateHis2File(username)
        self.API_CLASSIFIER_TRAIN(username)

class FEEDBACK:

    his = TEST_HistoricalDataGenerate()

    def API_UpdateRecords(self,username,boughtInfo,response_code,displayNum=30):
        #boughtInfo is a dataframe
        tablename = self.his.TEST_RESULT_PATH+'/'+ username+'_'+str(response_code) + '.csv'
        df_rec = pd.read_csv(tablename)
        df_rec = df_rec.head(displayNum)
        del(df_rec['Score'])
        if len(df_rec) < displayNum:
            BoughtTag = [0]*len(df_rec)
        else:
            BoughtTag = [0]*displayNum
        df_rec['Bought Tag'] = BoughtTag


        boughtID_list = boughtInfo['Travel Offer ID'].values.tolist()
        print(boughtID_list)
        for boughtID in boughtID_list:
            if  boughtID in df_rec['Travel Offer ID'].values.tolist():
                df_rec.drop(index=(df_rec.loc[(df_rec['Travel Offer ID']==boughtID)].index),inplace=True)
            else:
                pass
                #may fetch info from DB, or get info directly

        boughtInfo['Bought Tag'] = [1]*len(boughtInfo)
        df_res = pd.concat([df_rec,boughtInfo])
        df_res = df_res.reset_index(drop=True)
        cols=['Travel Offer ID','User ID','TimeStamp','Date Of Birth','city','country','Loyalty Card',
              'Payment Card','PRM Type','Preferred means of transportation','Preferred carrier',
              'Class','Seat','Refund Type','Quick','Reliable','Cheap','Comfortable','Door-to-door',
              'Environmentally friendly','Short','Multitasking','Social','Panoramic','Healthy',
              'Legs Number','Profile','Starting point','Destination','Via','LegMode','LegCarrier',
              'LegSeat','LegLength','Departure time','Arrival time','Services','Transfers','Transfer duration',
              'Walking distance to stop','Walking speed','Cycling distance to stop','Cycling speed',
              'Driving speed','Bought Tag']
        df_res = df_res.ix[:,cols]
        print('After Buying, the following historical records will be recorded:\n',df_res)

        df_res.to_csv(self.his.TEST_HISTORICAL_DATA_PATH + '/'+username+'.csv',mode='a',header=False,index=False)
        # self.dataP.renew_database(df_res, username)

    def get_boughtInfo(self,username,boughtList,response_code):
        RS_path = self.his.TEST_RESULT_PATH + '/'+username+'_'+ str(response_code)+'.csv'
        df_travel_offer = pd.read_csv(RS_path)
        del(df_travel_offer['Score'])
        boughtInfo = pd.DataFrame()

        for id in boughtList:
            df_tmp = df_travel_offer.loc[(df_travel_offer['Travel Offer ID']==id)]
            df_tmp['Bought Tag'] = 1
            boughtInfo = pd.concat([df_tmp,boughtInfo]) 

        boughtInfo = boughtInfo.reset_index(drop=True)
        return boughtInfo

    def CHECK_USER_FEEDBACK(self,username='TESTUSER1',boughtID_list=[2021043019424627],response_code=999):

        boughtInfo = self.get_boughtInfo(username, boughtID_list, response_code)
        self.API_UpdateRecords(username, boughtInfo, response_code)

#COLD USER PART
class CLUSTER_LEANER:

    his =TEST_HistoricalDataGenerate()
    ppl = DATA_Population()
    cluster = CLUSTER()
    classifier = CLASSIFIER_LEANER()
    

    #########################################################################################################
    ##########                                                                                     ##########
    ##########                         [CLUSTER]:FETCH TRAINING DATA FOR CLUSTER MODEL             ##########
    ##########                                                                                     ##########
    #########################################################################################################

    def update_userProfile_forCluster(self):
        
        AllUserProfile = pd.DataFrame()
        for user in os.listdir(self.his.TEST_USER_PROFILE):
            userprofile = pd.read_csv(self.his.TEST_USER_PROFILE+'/'+user)
            usercurr_profile = userprofile.sort_values(['TimeStamp'],ascending=False)[0:1]
            AllUserProfile = pd.concat([usercurr_profile,AllUserProfile])

        self.his.res2csv(AllUserProfile, self.cluster.CM_FOLDER_PATH, 'AllUserProfile.csv')

        return AllUserProfile

    def _get_TrainData_forCluster(self,allprofile_tablename='AllUserProfile.csv',updateTag=False):

        Profile_results = {}
        if updateTag is True:
            df_AllProfile = self.update_userProfile_forCluster()
        else:
            df_AllProfile = pd.read_csv(self.cluster.CM_FOLDER_PATH+'/'+allprofile_tablename)
        
        df_ProfileCopy = df_AllProfile.copy(deep=True)
        del(df_ProfileCopy['TimeStamp'])
        del(df_ProfileCopy['User ID'])
        df_onehot = self.ppl.data_OneHot(df_ProfileCopy)

        if len(df_onehot) > 1 :
            df_pure =self.ppl.data_Pure(df_onehot)
        else:
            df_pure = df_onehot

        feature_used = df_pure.columns.tolist()

        Profile_results.update({'Trained Profile':df_pure,'feature used':feature_used,'Orignal Profile':df_AllProfile})

        self.cluster.save_info(Profile_results, 'ClusterModelINFO','clusterM')

        return Profile_results

    #########################################################################################################
    ##########                                                                                     ##########
    ##########                            [CLUSTER]: PARAMETERS TUNNING                            ##########
    ##########                                                                                     ##########
    #########################################################################################################

    def API_Cluster_ParametersTunning(self,algorithm='DBSCAN',tunning_method=None,data=None):
        if data is None:
            res =self._get_TrainData_forCluster()
            data = res['Trained Profile']

        cluster_parameters = self.cluster.parameters_range_define(algorithm,tunning_method)
        tunning_result = self.cluster.cluster_parameters_tunning(data, cluster_parameters, algorithm,tunning_method=tunning_method)
        return tunning_result

    #########################################################################################################
    ##########                                                                                     ##########
    ##########                            [CLUSTER] MODEL TRAINING                                 ##########
    ##########                                                                                     ##########
    #########################################################################################################    

    def API_CLUSTER_TRAINING(self,algorithm='DBSCAN',data=None,original_userprofile=None,tunning_method=None,saveTag=True):
        if data is None :
            res = self._get_TrainData_forCluster()
            data = res['Trained Profile']
            original_userprofile = res ['Orignal Profile']

        best_parameter = self.API_Cluster_ParametersTunning(algorithm)
        model_info = self.cluster.cluster_model_fit(data, algorithm, best_parameter)
        self.cluster.save_info(model_info, '{}_CLUSTER_MODEL_INFO'.format(algorithm))
        if saveTag is True:
            self.API_ClusterModel_Builder(original_userprofile, model_info)

    #########################################################################################################
    ##########                                                                                     ##########
    ##########                            [CLUSTER]: BUILD MODEL FOR EACH CLUSTER                  ##########
    ##########                                                                                     ##########
    #########################################################################################################
    def API_ClusterModel_Builder(self,original_userprofile,model_info):
        cluster_res = self.cluster.get_fitUserDistribution(original_userprofile, model_info)

        #CREATE/UPDATE CLUSTER HISTORICAL TABLE IN HISTORICAL DB
        for clusterNo in cluster_res.keys():
            # self.dataP.Update_HistoriyDB_forCluster(cluster_res[clusterNo], clusterNo)
            tablename = 'Cluster_{}_Records.csv'.format((clusterNo))
            df_raw = pd.DataFrame()
            for user in cluster_res[clusterNo]:
                filename = user + '.csv'
                if filename in os.listdir(self.his.TEST_HISTORICAL_DATA_PATH):
                    df_raw_tmp = pd.read_csv(self.his.TEST_HISTORICAL_DATA_PATH+'/'+filename)
                    df_raw = pd.concat([df_raw,df_raw_tmp],axis=0)
            if len(df_raw) == 0:
                print('please check your database, one cluster has no data')
                # TODO:give a top popular travel offer list? problem will be solved when we have a big db.
                return 
            self.his.res2csv(df_raw, self.his.TEST_HISTORICAL_DATA_PATH, tablename)

        print('All the historical data for each cluster is ready!') 

        #TRAIN EACH CLUSTER DATA TO GET CLUSTER's MODEL
        for clusterNo in cluster_res.keys():
            tablename = 'Cluster_{}_Records'.format((clusterNo))
            recommender_categories = [
                            'KNeighborsClassifier',
                            # 'SVC',
                            'DecisionTreeClassifier',
                            'LogisticRegression',
                            'RandomForestClassifier'
            ]
            self.classifier.API_CLASSIFIER_TRAIN(tablename,recommender_category=recommender_categories,file_path='cluster') 

    def CHECK_ADMINISTRATOR2CLUSTER_TRAIN(self):
        self.update_userProfile_forCluster()
        self.API_CLUSTER_TRAINING()

    # def CHECK_ADMINISTRATOR2UPDATE

# his = TEST_HistoricalDataGenerate()
# his._GenerateHis2File('COLDUSER1')
# print(his.TEST_HISTORICAL_DATA_PATH)

# c = CLASSIFIER_LEANER()
# c._get_TrainData('TESTUSER1')
# c.CHECK_ADMINISTRATOR2CLASSIFIER_TRAIN('OLDUSER1')
# print(bst)

# clu = CLUSTER_LEANER()
# clu.CHECK_ADMINISTRATOR2CLUSTER_TRAIN()

# fdb = FEEDBACK()
# bought = [2021051201545715]
# res ='20210512_01545786'
# boughtInfo = fdb.get_boughtInfo('OLDUSER1',bought,res)
# fdb.API_UpdateRecords('OLDUSER1',boughtInfo,res)
