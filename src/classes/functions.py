# This class is Tool box
import numpy as np
import matplotlib.pyplot as plt
from configuration import configuration as config


class functions:
    def getRPS(data):
        '''
        This function is aimed to get the rotational speed (rps): r/s
        Input of this function: data from hall sensor (in form of str)
        Output of this function: rotational speed of the tool in each time point
        power supply for hall-sensor must be 5V
        '''
        ###### 1. get difference of point_{i+1}-point_i
        numRow = data.shape[0]-1                               # number of rows of hall-sensor data mines 1
        difference = np.zeros((numRow, 2))                     # point_{i+1}-point_i
        difference[:, 0] = data[0:-1, 0]
        difference[:, 1] = abs(data[1:, 1] - data[0:-1, 1])    # point_{i+1}-point_i
        
        ###### 2. Filter 1
        for i in range(numRow):
            if difference[i, 1] >= 0.2:
                difference[i, 1] = 1
            else:
                difference[i, 1] = 0
        
        ###### 3. Filter 2 for Gang 1                           # actually, maximal rps for gang 1 is 10, i will take 12
        for i in range(numRow-int(config.SF/12)):
            if difference[i, 1] == 1 and np.sum(difference[i:(i+int(config.SF/12)), 1])>= 1:
                difference[i, 1] = 1
                difference[(i+1):(i+int(config.SF/12)), 1] = 0
                
        ###### 4. get rps
        rps = np.zeros(difference.shape)                        # rps: rotational speed per second
        rps[:, 0] = difference[:, 0]
        SF4rps=int(config.SF/5)                                     
        for i in range(numRow):
            if i <SF4rps:
                rps[i,1] = 5*np.sum(difference[0:SF4rps,1]==1)
            if i < numRow-SF4rps and i >= SF4rps:
                rps[i,1] = 5*np.sum(difference[int(i-SF4rps/2):int(i+SF4rps/2),1]==1)
            if i >=numRow-SF4rps:
                rps[i,1] = 5*np.sum(difference[(i-SF4rps):,1]==1)
        
        ###### 5. Filter 3
        rps = functions.meanFLT(rps, 250)                       # use a mean value filter t make it smooth
        
        ###### 7. for check
        # plt.figure()
        # plt.plot(difference[:,0],difference[:,1],'b')
        # plt.xlabel('Zeit / [s]',fontsize=16)
        # plt.ylabel('rps / [r/s]',fontsize=16)
        # plt.title('Difference at different time point',fontsize=16)
        # plt.show()  
        # 
        # plt.figure()
        # plt.plot(rps[:,0],rps[:,1],'b')
        # plt.xlabel('Zeit / [s]',fontsize=16)
        # plt.ylabel('rps / [r/s]',fontsize=16)
        # plt.title('Rotational speed at different time point',fontsize=16)
        # plt.show()   
        
        return rps
    



    def meanFLT(data, numP4meanValFLT):   
        # calculate the mean value of every 2*numP4meanValFL+1 points
        meanData = np.zeros(data.shape)                                 # mean value                                                                  
        meanData[:, 0] = data[:, 0]                                                                                         # time axis
        for k in range(numP4meanValFLT):                                                                                                  
            meanData[k,1:] = sum(data[0:(k+numP4meanValFLT+1),1:])/(k+numP4meanValFLT+1)                                    # deal with the "head"
            meanData[-1-k,1:] = sum(data[-(k+numP4meanValFLT+1):,1:])/(k+numP4meanValFLT+1)                                 # deal with the "tail"
                             
        meanData[numP4meanValFLT,1:] = sum(data[0:(2*numP4meanValFLT+1),1:])/(2*numP4meanValFLT+1)                          # deal with the first point of the "middle"
        tempVar = meanData[numP4meanValFLT,1:]                                                                              # Temporary variable
        for kk in range(numP4meanValFLT+1, (data.shape[0] - numP4meanValFLT), 1):                                           # deal with the "middle"
            meanData[kk,1:] = tempVar + (data[kk+numP4meanValFLT,1:] - data[kk-numP4meanValFLT-1,1:])/(2*numP4meanValFLT+1) # deal with the "middle"
            tempVar = meanData[kk, 1:]
        
        return meanData