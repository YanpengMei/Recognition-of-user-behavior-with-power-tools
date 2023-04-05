# This class is aimed to read and process the data
from multiprocessing import Pool
import numpy as np
import pandas as pd
import openpyxl
from classes.functions import functions as fun
from configuration import configuration as config


class preProcessing:
    def prePro_main(mainPath):
        # Abbreviations of sensors: GYRO_X,GYRO_Y,GYRO_Z,ACC_X,ACC_Y,ACC_Z,VLT,CURT,HALL(for hall sensor),FON(for mikrofon)
        # Numbers, that are representing sensors:GYRO_X,GYRO_Y,GYRO_Z,ACC_X,ACC_Y,ACC_Z,VLT,CURT,HALL,FON --> 1,2,3,4,5,6,7,8,9,10

        ###### 1. get path for each csv-file
        path4GYRO_X = mainPath + "\csvNI\GYRO_X.csv"
        path4GYRO_Y = mainPath + "\csvNI\GYRO_Y.csv"
        path4GYRO_Z = mainPath + "\csvNI\GYRO_Z.csv"
        path4ACC_X = mainPath + "\csvNI\ACC_X.csv"
        path4ACC_Y = mainPath + "\csvNI\ACC_Y.csv"
        path4ACC_Z = mainPath + "\csvNI\ACC_Z.csv"
        path4VLT = mainPath + "\csvNI\VLT.csv"
        path4CURT = mainPath + "\csvNI\CURT.csv"
        path4HALL = mainPath + "\csvNI\HALL.csv"
        path4FON = mainPath + "\csvNI\FON.csv"

        ###### 2. read csv-file with 6 processors
        inputList = [(path4GYRO_X,1),(path4GYRO_Y,2),(path4GYRO_Z,3),(path4ACC_X,4),(path4ACC_Y,5),(path4ACC_Z,6),(path4VLT,7),(path4CURT,8),(path4HALL,9),(path4FON,10)]
        pool = Pool(6)
        outputList = pool.map(preProcessing.readCSV, inputList)
        pool.close()
        pool.join()

        ###### 3. get allSensorData, allSensorData = (GYRO_X,GYRO_Y,GYRO_Z,ACC_X,ACC_Y,ACC_Z,VLT,CURT,HALL,FON)
        allSensorData = ()
        for i in range(config.numSS):
            numP = outputList[i][0].shape[0]                                             # number of points
            dataTemp = np.zeros((numP, 1))
            for j in range(config.numSS):
                # GYRO_X
                if i == 0 and outputList[j][1] == 1:
                    dataTemp[:, 0] = outputList[j][0][:, 1]
                    allSensorData = allSensorData + (dataTemp,)
                # GYRO_Y
                if i == 1 and outputList[j][1] == 2:
                    dataTemp[:, 0] = outputList[j][0][:, 1]
                    allSensorData = allSensorData + (dataTemp,)
                # GYRO_Z
                if i == 2 and outputList[j][1] == 3:
                    dataTemp[:, 0] = outputList[j][0][:, 1]
                    allSensorData = allSensorData + (dataTemp,)
                # ACC_X
                if i == 3 and outputList[j][1] == 4:
                    dataTemp[:, 0] = outputList[j][0][:, 1]
                    allSensorData = allSensorData + (dataTemp,)
                # ACC_Y
                if i == 4 and outputList[j][1] == 5:
                    dataTemp[:, 0] = outputList[j][0][:, 1]
                    allSensorData = allSensorData + (dataTemp,)
                # ACC_Z
                if i == 5 and outputList[j][1] == 6:
                    dataTemp[:, 0] = outputList[j][0][:, 1]
                    allSensorData = allSensorData + (dataTemp,)
                # VLT
                if i == 6 and outputList[j][1] == 7:
                    dataTemp[:, 0] = outputList[j][0][:, 1]
                    allSensorData = allSensorData + (dataTemp,)
                # CURT
                if i == 7 and outputList[j][1] == 8:
                    dataTemp[:, 0] = outputList[j][0][:, 1]
                    allSensorData = allSensorData + (dataTemp,)
                # HALL
                if i == 8 and outputList[j][1] == 9:
                    dataTemp[:, 0] = outputList[j][0][:, 1]
                    allSensorData = allSensorData + (dataTemp,)
                # FON
                if i == 9 and outputList[j][1] == 10:
                    dataTemp[:, 0] = outputList[j][0][:, 1]
                    allSensorData = allSensorData + (dataTemp,)
        return allSensorData




    def readCSV(inputTuple):
        # This function is aimed to read the .csv-files                                   
        data_str = np.loadtxt(open(inputTuple[0], 'r'), dtype='str_', delimiter=';',skiprows=100)
        data = np.zeros(data_str.shape)
        for i in range(data_str.shape[0]):
            data[i,0] = float(np.char.replace(data_str[i,0],',','.'))
            if inputTuple[1] == 8:
                data[i,1] = 1000*float(np.char.replace(data_str[i,1],',','.'))          # For CURT, there is a scaling 1/1000 during the measurement
            if inputTuple[1] == 7:
                data[i,1] = 100*float(np.char.replace(data_str[i,1],',','.'))           # For VLT, there is a scaling 1/100 during the measurement
            if inputTuple[1] == 1 or inputTuple[1] == 2 or inputTuple[1] == 3:
                data[i,1] = float(np.char.replace(data_str[i,1],',','.'))-1.67          # For GYRO, there is a Offset +1.67 V during the measurement
            else:
                data[i,1] = float(np.char.replace(data_str[i,1],',','.'))               # For other sensors, there is no scaling during the measurement
        if inputTuple[1]==9:                                                            # For hall-sensor, we will convert the hall-signal to rotation speed
            data = fun.getRPS(data)
        
        return data, inputTuple[1]




    def segmentation(mainPath, allSensorData):
        # This script is aimed to segment all the sensor data according to the current data
        ###### 1. get the index of numpy array from excel-table, to segment the data
        pathExcel = mainPath + "\idx.xlsx"
        file = openpyxl.load_workbook(pathExcel)
        table = file.worksheets[0]
        idx_tuple1 = list(table.columns)[1]
        idx_tuple2 = list(table.columns)[2]
        idx = np.zeros((1,int(2*len(idx_tuple1))))
        i = 0
        for cell in idx_tuple1:
            idx[0,2*i] = cell.value
            i=i+1 
        i = 0
        for cell in idx_tuple2:
            idx[0,2*i+1] = cell.value
            i=i+1                                                 
        
        idx_list = idx[0,:].tolist()
        idx_list = list(map(int,idx_list))
        # print(len(idx_list),idx_list)
        
        ###### 2. segment the data
        allSensorData_seg = ()
        for i in range(config.numSS):
            sensor_i = ()
            for j in range(int(len(idx_list) / 2)):
                sensor_i = sensor_i + (allSensorData[i][int(idx_list[2*j]):int(idx_list[2*j+1]),0],)
            allSensorData_seg = allSensorData_seg + (sensor_i,)
        return allSensorData_seg