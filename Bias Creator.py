import csv
import math
import numpy as np

TOTAL_DATA_PTS = 1200
TOTAL_TEST_PTS = 0
NUMBER_PTS_GRAB = 1000
NUMBER_PTS_TRAIN = 800
PERCENTAGE_BIAS = 0.1
BIAS_LOCATION_NUM = 2
ITERATIONS = 4

DISTRIBUTION_TALBES = ()
TABLE_LIST = []
CSV_TABLE = []
CSV_FILE = 'seattle-weather.csv'

REMAINDER_LIST = []

class Distribution:
    def __init__(self, data):
        dataset = [float(d) for d in data]
        self.mean = 0
        self.stdDev = 0
        self.max = max(dataset)
        self.min = min(dataset)
    def getRandomNum(self):
        rand = round(np.random.uniform(low=self.min, high = self.max),3)
        return rand

class Selection_Bias:
    def __init__(self, biasNum, table):
        self.biasNumber = biasNum
        self.distTable = {}
        self.biasTable = []
        for col in table:
            self.setColumns(col)

    def calculateDate(self, date):
        year,month,day = date.split('-')
        return (int(day) + int(month) * 31 + int(year) * 31 * 12)

    def reverseCalculateDate(self, num):
        num = round(num)
        year = 2000
        while (year+1) *31 *12 < num:
            year += 1
        num = num - (year * 31 * 12)
        month = 1
        while (month+1) *31 < num:
            month += 1
        num = num - (month*31)
        day = num
        return '{}-{}-{}'.format(year,month,day)

    def setColumns(self, col):
        name = col[0]
        match name:
            case 'date':
                dateCol = []
                for c in col[1:]:
                    dateCol.append(self.calculateDate(c))
                distr = Distribution(dateCol)
                self.distTable[name] = distr
            case _:
                dataCol = col[1:]
                distr = Distribution(dataCol)
                self.distTable[name] = distr
                
    def setBiases(self):
        keyset = list(self.distTable)
        for _ in range(self.biasNumber):
            var = {}
            for name in keyset:
                if name == 'date':
                    randNum = round(self.distTable[name].getRandomNum(),2)
                    date = self.reverseCalculateDate(randNum)
                    var[name] = (date, randNum)
                else:
                    var[name] = round(self.distTable[name].getRandomNum(),2)
            self.biasTable.append(var)
    #def select_nearest():
    def printBias(self):
        for temp in self.biasTable:
            print(temp)

    def findNearest(self,table,num):
        length = len(table[0])
        categories = len(table)
        split = int(num/self.biasNumber)
        nearestList = []
        count = [0 for i in range(self.biasNumber)]
        for group in range(self.biasNumber):
            nearArray = []
            for row in range(1,length):
                total = 0
                for cat in range(categories):
                    name = table[cat][0]
                    if cat == 0: #for date
                        val = self.calculateDate(table[cat][row])
                    else:
                        val = float(table[cat][row])
                        total += pow(val - self.biasTable[group][name],2)
                total = math.sqrt(total)
                nearArray.append([total,row])
            #print(nearArray[:20])
            #nearestList.append( sorted(nearArray, key=lambda x: x[0])[:split] )
            nearArray.sort(key=lambda x: x[0])
            nearestList.extend( [ x[1] for x in nearArray[:split] ] )
        return nearestList
            
            
def setTableList():
    global TABLE_LIST
    global CSV_TABLE
    global REMAINDER_LIST
    global TOTAL_DATA_PTS
    with open(CSV_FILE) as csvfile:
        reader = csv.reader(csvfile)
        start = True
        count = 0
        
        for row in reader:
            if count > TOTAL_DATA_PTS:
                break
            if count <= NUMBER_PTS_GRAB:
                if start: # intitalizes table
                    start = False
                    TABLE_LIST = [[] for i in range(len(row)-1)]
                    REMAINDER_LIST.append([i for i in row])
                for i in range(len(row)-1):
                    TABLE_LIST[i].append(row[i])
                CSV_TABLE.append(row)
            else:
                REMAINDER_LIST.append(row)
            count += 1

def main():
    global TABLE_LIST
    global PERCENTAGE_BIAS
    global ITERATIONS
    global REMAINDER_LIST
    global CSV_TABLE
    global NUMBER_PTS_TRAIN
    
    setTableList()
    first = True
    percentageBiasList = []
    firstRow = CSV_TABLE[0]

    distribution = Selection_Bias(BIAS_LOCATION_NUM, TABLE_LIST)
    distribution.setBiases()
    distribution.printBias()
    
    for iterations in range(ITERATIONS + 1): #making 30 files and one control (0.0 bias)
        if first:
            PERCENTAGE_BIAS = 0.0
        else:
            #PERCENTAGE_BIAS = round(np.random.uniform(low=0.01, high=0.5),3)
            PERCENTAGE_BIAS += 0.1
        if not first:
            percentageBiasList.append(PERCENTAGE_BIAS)

        # randomly create biases
        biasNum = int(NUMBER_PTS_TRAIN * PERCENTAGE_BIAS)
        rest = NUMBER_PTS_TRAIN - biasNum
        print('number of biased points: ' + str(biasNum))
        biasList = [i for i in range(1,rest+1)]
        biasList.extend(distribution.findNearest(TABLE_LIST, biasNum)) # grab bias list
        inputList = [CSV_TABLE[index] for index in biasList]
        print(len(inputList))
        
        filename = ""
        if first: # first file is the control file
            first = False
            filename = "control_file.csv"
        else:
            filename = "bias_file" + str(iterations) + ".csv"
        
        with open(filename,"w",newline='') as my_csv:
            csvWriter = csv.writer(my_csv)
            csvWriter.writerow(firstRow)
            csvWriter.writerows(inputList)
    print(percentageBiasList)

    #stores remaining percentages
    with open("percentage.csv","w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv)
        for item in percentageBiasList:
            csvWriter.writerow([item])

    with open("Test_List.csv","w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv)
        for item in REMAINDER_LIST:
            csvWriter.writerow(item)

if __name__ == "__main__":
    main()
