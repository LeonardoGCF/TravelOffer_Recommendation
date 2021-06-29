import os
from numpy.lib.function_base import average
import pandas as pd
from numpy.lib.utils import info
from LEARNER import TEST_HistoricalDataGenerate
from LEARNER import CLASSIFIER_LEANER
from ParametersTunning import ParametersTunning

import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

class ACC_EVAL:
    his = TEST_HistoricalDataGenerate()
    classifier =CLASSIFIER_LEANER()
    PT =ParametersTunning()

    def generate_Eval_Dataset(self,dataName,dataCount):
        for n in range(dataCount):
            username = dataName+'_'+str(n)
            self.his._GenerateHis2File(username)
        print('Evaluation Dataset of {} users has already generated!'.format(dataName))
        return dataName

    def collect_Accuracy_info(self,username,reTrainTag=True,fit_data_tag='all',model_version='latest'):
        if reTrainTag == True:
            self.classifier.API_CLASSIFIER_TRAIN(username,fit_data_tag=fit_data_tag)
        save_file_name = 'best_model_{}_{}_{}'.format(username,fit_data_tag,model_version)
        infoDict = self.PT.load_info(save_file_name)
        # print(infoDict)
        return infoDict

    def evaluation_results(self,dataName,count,reTrainTag=True):
        dataframe_lists = []
        
        for n in range(count):
            evalDict ={}
            dataframe_list=[]
            username = dataName +'_'+ str(n)
            info_tmp = self.collect_Accuracy_info(username,reTrainTag)
            evalDict.update({'username':username})

            if info_tmp['KNeighborsClassifier_uniform']['AccScore'] >= info_tmp['KNeighborsClassifier_distance']['AccScore']:
                evalDict.update({'KNN_Acc':info_tmp['KNeighborsClassifier_uniform']['AccScore'],
                'KNN_timeconsuming':info_tmp['KNeighborsClassifier_uniform']['Timeconsuming']})
                knn_trainingtime = info_tmp['KNeighborsClassifier_uniform']['Timeconsuming'] 
            elif info_tmp['KNeighborsClassifier_uniform']['AccScore'] < info_tmp['KNeighborsClassifier_distance']['AccScore']:
                evalDict.update({'KNN_Acc':info_tmp['KNeighborsClassifier_distance']['AccScore'],
                'KNN_timeconsuming':info_tmp['KNeighborsClassifier_distance']['Timeconsuming']}) 
                knn_trainingtime = info_tmp['KNeighborsClassifier_distance']['Timeconsuming']  

            evalDict.update({
                'SVC_Acc':info_tmp['SVC']['AccScore'],
                'SVC_timeconsuming':info_tmp['SVC']['Timeconsuming'],
                'DecisionTree_Acc':info_tmp['DecisionTreeClassifier']['AccScore'],
                'DecisionTree_timeconsuming':info_tmp['DecisionTreeClassifier']['Timeconsuming'],
                'LogisticRegression_Acc':info_tmp['LogisticRegression']['AccScore'],
                'LogisticRegression_timeconsuming':info_tmp['LogisticRegression']['Timeconsuming'],
                'RandomForest_Acc':info_tmp['RandomForestClassifier']['AccScore'],
                'RandomForest_timeconsuming':info_tmp['RandomForestClassifier']['Timeconsuming'],
                'BEST_MODEL':info_tmp['best_recommender'],
                'BEST_Acc':info_tmp['best_score'],
                'BEST_MODEL_timeconsuming':info_tmp['best_timeconsuming']
            })

            total_trainingtime = knn_trainingtime + info_tmp['SVC']['Timeconsuming'] + info_tmp['DecisionTreeClassifier']['Timeconsuming']+ info_tmp['LogisticRegression']['AccScore'] + info_tmp['RandomForestClassifier']['Timeconsuming']
            evalDict.update({'Total_trainingtime':total_trainingtime})

            dataframe_list.append(evalDict['username'])
            dataframe_list.append(evalDict["KNN_Acc"])
            dataframe_list.append(evalDict["KNN_timeconsuming"])
            dataframe_list.append(evalDict["SVC_Acc"])
            dataframe_list.append(evalDict["SVC_timeconsuming"])   
            dataframe_list.append(evalDict["DecisionTree_Acc"])
            dataframe_list.append(evalDict["DecisionTree_timeconsuming"])
            dataframe_list.append(evalDict["LogisticRegression_Acc"])
            dataframe_list.append(evalDict["LogisticRegression_timeconsuming"])
            dataframe_list.append(evalDict["RandomForest_Acc"])
            dataframe_list.append(evalDict["RandomForest_timeconsuming"])
            dataframe_list.append(evalDict["BEST_MODEL"])
            dataframe_list.append(evalDict["BEST_Acc"])
            dataframe_list.append(evalDict["BEST_MODEL_timeconsuming"])
            dataframe_list.append(evalDict['Total_trainingtime'])

            dataframe_lists.append(dataframe_list)

        df = pd.DataFrame(data=dataframe_lists)
        df.columns =['username',
                    "KNN_Acc",
                    "KNN_trainingtime",
                    "SVC_Acc",
                    "SVC_trainingtime",   
                    "DecisionTree_Acc",
                    "DecisionTree_trainingtime",
                    "LogisticRegression_Acc",
                    "LogisticRegression_trainingtime",
                    "RandomForest_Acc",
                    "RandomForest_trainingtime",
                    "BEST_MODEL",
                    "BEST_Acc",
                    "BEST_MODEL_trainingtime",
                    'Total_trainingtime']
        
        return df

    def res2csv(self,df,file_path,filename):
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        filename = file_path +'/'+filename 
        df.to_csv(filename,encoding='utf-8',mode='w',header=True,index=False,float_format='%.5f')


    def _run(self,dataName ='AccTestUser',count = 1000,file_path=None,filename='AccEvalResult.csv',reGenerateDataTag=True,reTrainTag=True):
        if reGenerateDataTag ==True:
            self.generate_Eval_Dataset(dataName,count)
        df = self.evaluation_results(dataName,count,reTrainTag)
        if file_path ==None:
            file_path =self.his.EVAL_ACC_RESULT_PATH
        self.res2csv(df,file_path,filename)
        print('Result of accuracy evaluation has already generated successfully in :',file_path+'/'+filename)

    def csv_Analysis(self,filename='AccEvalResult.csv',filepath=None):
        if filepath == None:
            filepath=self.his.EVAL_ACC_RESULT_PATH
        res_Full = pd.read_csv(filepath+'/'+filename)

        #Separate Box plot
        res_Acc = res_Full[['KNN_Acc','SVC_Acc','DecisionTree_Acc','LogisticRegression_Acc','RandomForest_Acc']]#,'BEST_Acc']]
        # print(res_Acc)
        data_Box = res_Acc.values
        labels=['KNN_Acc','SVC_Acc','DecisionTree_Acc','LogisticRegression_Acc','RandomForest_Acc'] #,'BEST_Acc']

        plt.boxplot(data_Box,labels=labels)
        plt.title('Accuracy Boxplot of Different Algorithms for User "AccTestUser_0"-"AccTestUser_999"')
        plt.show()

        #Whole Box plot
        best_Acc = res_Full[['BEST_Acc']]
        # print(res_Acc)
        best_Box = best_Acc.values
        best_labels=['BEST_Acc']

        plt.boxplot(best_Box,labels=best_labels)
        plt.title('Accuracy Boxplot of the Best Model for User "AccTestUser_0"-"AccTestUser_999"')
        plt.show()

        #Pie Chart
        pie_arr = res_Full['BEST_MODEL']
        print(pie_arr)
        pie_values =[0]*5
        for n in range(len(pie_arr)):
            if pie_arr[n] == 'KNeighborsClassifier_uniform' or pie_arr[n]=='KNeighborsClassifier_distance':
                pie_values[0] += 1
            elif pie_arr[n] == 'SVC':
                pie_values[1] +=1
            elif pie_arr[n] == 'DecisionTreeClassifier':
                pie_values[2] +=1
            elif pie_arr[n] == 'LogisticRegression':
                pie_values[3] +=1
            elif pie_arr[n] == 'RandomForestClassifier':
                pie_values[4] +=1
        
        pie_values = [x/len(pie_arr) for x in pie_values]
        pie_labels =['KNN','SVC','Decision Tree','Logistic Regression','Random Forest']
        plt.pie(x=pie_values,labels=pie_labels,autopct='%1.2f%%')
        plt.title('The probability of different algorithms that are used to generate the best model.')
        plt.show()

        #Training time
        #Separate Box plot
        res_Time = res_Full[['KNN_trainingtime','SVC_trainingtime','DecisionTree_trainingtime','LogisticRegression_trainingtime','RandomForest_trainingtime']]#,'BEST_trainingtime']]
        # print(res_Acc)
        time_Box = res_Time.values
        time_labels=['KNN_trainingtime','SVC_trainingtime','DecisionTree_trainingtime','LogisticRegression_trainingtime','RandomForest_trainingtime'] #,'BEST_trainingtime']

        plt.boxplot(time_Box,labels=time_labels)
        plt.title('Training time Boxplot of Different Algorithms for User "AccTestUser_0"-"AccTestUser_999"')
        plt.show() 

        #Whole Box plot
        total_Time = res_Full['Total_trainingtime']
        # print(res_Acc)
        totaltime_Box = total_Time
        totaltime_labels=['Total Training Time']

        plt.boxplot(totaltime_Box,labels=totaltime_labels)
        plt.title('Total Training Time Boxplot for User "AccTestUser_0"-"AccTestUser_999"')
        plt.show()

main =ACC_EVAL()
dataName ='AccTestUser'
count = 1000
# main.generate_Eval_Dataset(dataName,count)

# info = main.collect_Accuracy_info(dataName +'_0')
# print(info['SVC'])
# print(info['SVC']['AccScore'])

# df = main.evaluation_results(dataName,count)
# print(df)

# main._run(reGenerateDataTag=False,reTrainTag=True)
# main._run()

# for n in range(94,1000):
#     main.collect_Accuracy_info(dataName +'_'+str(n))

main.csv_Analysis()