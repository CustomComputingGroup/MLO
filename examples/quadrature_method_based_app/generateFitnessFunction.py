#python data dict creator
#
#AnsonExec.csv
#text delimiter " field delimiter  ,
#
#to get number of cores
#AnsonCores.csv
import csv
global allData
allData=None
def getAllData():
    global allData
    if not allData:
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