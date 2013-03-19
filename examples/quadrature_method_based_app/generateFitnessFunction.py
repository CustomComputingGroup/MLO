#python data dict creator
#
#AnsonExec.csv
#text delimiter " field delimiter  ,
#
#to get number of cores
#AnsonCores.csv
import csv
from numpy import array
from sklearn.gaussian_process import GaussianProcess
from sklearn import preprocessing

import random
global allData, costModel, costModelInputScaler, costModelOutputScaler
allData=None
costModel=None
costModelInputScaler=None
costModelOutputScaler=None
def getAllData():
    global allData, costModel, costModelInputScaler, costModelOutputScaler
    if not allData or not costModel:
        ## COST MODEL
        spamReader = csv.reader(open('time_results.csv', 'rb'), delimiter=';', quotechar='"')

        x = []
        y = []
        for row in spamReader:
            x.append([float(row[1]),float(row[2])])
            y.append([float(row[3]) + random.random()])
        x = array(x)
        y = array(y)
        input_scaler = preprocessing.StandardScaler().fit(x)
        scaled_training_set = input_scaler.transform(x)

                # Scale training data
        output_scaler = preprocessing.StandardScaler(with_std=False).fit(y)
        adjusted_training_fitness = output_scaler.transform(y)
        
        regr = GaussianProcess(corr='squared_exponential', theta0=1e-1,
                         thetaL=1e-5, thetaU=3,
                         random_start=400)
        regr.fit(scaled_training_set, adjusted_training_fitness)
        costModel = regr
        costModelInputScaler = input_scaler
        costModelOutputScaler = output_scaler
        ## cores, accuracy, exeuction time
        spamReader = csv.reader(open('AnsonCores.csv', 'rb'), delimiter=',', quotechar='"')
        cores = {11:{}}
        for row in spamReader:
            cores[11][int(row[1])] = int(row[0])

        maxcores = cores
        spamReader = csv.reader(open('AnsonExec.csv', 'rb'), delimiter=';', quotechar='"')

        allData = {}
        for row in spamReader:
            row_0 = int(row[0])
            row_1 = int(row[1])
            row_2 = int(row[2])
            row_3 = float(row[3])
            row_4 = float(row[4])
            data = [cores[row_0][row_1],row_3,row_4]
            
            try:
                try:
                    allData[row_0][row_1][row_2] = data
                except:
                    allData[row_0][row_1] = {row_2:data}
            except:
                allData[row_0] = {row_1:{row_2:data}}
        #spamReader.close()
    #print allData
    return allData
#print allData
#print cores

def getCost(wF, cores):
    global costModel, costModelInputScaler, costModelOutputScaler
    return costModelOutputScaler.inverse_transform(costModel.predict(costModelInputScaler.transform(array([[wF,cores]]))))
    #print regr.predict(array([[11,41,8]]))
    