# This script is to see the importance of each sensor
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold
from configuration import configuration as config


def mainScript():
    ###### 1. get path and allFeatures
    path = os.path.dirname(os.getcwd()) + '/data/Test021bis008'         # main path
    path4features = path + '/allFeatures.csv'                           # path of the file with allFeatures
    allFeatures = np.array(pd.read_csv(path4features, header=None))     # read all features from allFeatures.csv

    ###### 2. normalization of allFeatures
    normalize = StandardScaler()                                        
    allFeatures_N = normalize.fit_transform(allFeatures)                # normalized features

    ###### 3. Add labels y 
    numTests = 7                                                        # data in test 8,9,10,11,12,13,18 will be used
    y = np.zeros((numTests*config.numUB*config.numDS,1))
    for j in range(numTests):
        for i in range(config.numUB): 
            if i == (config.numUB-1) and j == (numTests - 1):
                y[(j*config.numUB*config.numDS + i*config.numDS):] = i+1
            else:
                y[(j*config.numUB*config.numDS + i*config.numDS):(j*config.numUB*config.numDS + (i+1)*config.numDS)] = i+1
    
    
    
    
    ###### 4. Evaluation 1: which sensor datas are necessary
    '''
    ### 4.1 with all sensor-datas
    # x are features, that will be used for training and testing
    x = np.zeros((numTests*config.numUB*config.numDS, config.numFeatures*config.numSS))                                                         # x:features, y:labels
    x[0:6*config.numUB*config.numDS, :] = allFeatures_N[0:6*config.numUB*config.numDS, :]
    x[6*config.numUB*config.numDS:, :] = allFeatures_N[8*config.numUB*config.numDS:9*config.numUB*config.numDS, :]
    '''

    #'''
    ### 4.2 without GYRO-data                             
    x = np.zeros((numTests*config.numUB*config.numDS, config.numFeatures*(config.numSS-3)))                                                         # x:features, y:labels
    x[0:6*config.numUB*config.numDS, :] = allFeatures_N[0:6*config.numUB*config.numDS, config.numFeatures*3:]
    x[6*config.numUB*config.numDS:, :] = allFeatures_N[8*config.numUB*config.numDS:9*config.numUB*config.numDS, config.numFeatures*3:]
    #'''

    '''
    ### 4.3 without GYRO-data & ACC-data                              
    x = np.zeros((numTests*config.numUB*config.numDS,config.numFeatures*(config.numSS-6)))                                                         # x:features, y:labels
    x[0:6*config.numUB*config.numDS, :] = allFeatures_N[0:6*config.numUB*config.numDS,config.numFeatures*6:]
    x[6*config.numUB*config.numDS:, :] = allFeatures_N[8*config.numUB*config.numDS:9*config.numUB*config.numDS, config.numFeatures*6:]
    '''
    
    '''
    ### 4.4 without GYRO-data & ACC-data & VLT-data                             
    x = np.zeros((numTests*config.numUB*config.numDS,config.numFeatures*(config.numSS-7)))                                                         # x:features, y:labels
    x[0:6*config.numUB*config.numDS, :] = allFeatures_N[0:6*config.numUB*config.numDS,config.numFeatures*7:]
    x[6*config.numUB*config.numDS:, :] = allFeatures_N[8*config.numUB*config.numDS:9*config.numUB*config.numDS, config.numFeatures*7:]
    '''

    '''
    ### 4.5 without GYRO-data & ACC-data & VLT-data & CURT-data                             
    x = np.zeros((numTests*config.numUB*config.numDS,config.numFeatures*(config.numSS-8)))                                                         # x:features, y:labels
    x[0:6*config.numUB*config.numDS, :] = allFeatures_N[0:6*config.numUB*config.numDS,config.numFeatures*8:]
    x[6*config.numUB*config.numDS:, :] = allFeatures_N[8*config.numUB*config.numDS:9*config.numUB*config.numDS, config.numFeatures*8:]
    '''

    '''
    ### 4.6 without GYRO-data & ACC-data & VLT-data & FON-data                          
    x = np.zeros((numTests*config.numUB*config.numDS,config.numFeatures*(config.numSS-8)))                                                         # x:features, y:labels
    x[0:6*config.numUB*config.numDS, 0:config.numFeatures] = allFeatures_N[0:6*config.numUB*config.numDS, config.numFeatures*7:config.numFeatures*8]
    x[0:6*config.numUB*config.numDS, config.numFeatures:] = allFeatures_N[0:6*config.numUB*config.numDS,config.numFeatures*9:]
    x[6*config.numUB*config.numDS:, 0:config.numFeatures] = allFeatures_N[8*config.numUB*config.numDS:9*config.numUB*config.numDS, config.numFeatures*7:config.numFeatures*8]
    x[6*config.numUB*config.numDS:, config.numFeatures:] = allFeatures_N[8*config.numUB*config.numDS:9*config.numUB*config.numDS, config.numFeatures*9:]
    '''

    '''
    ### 4.7 without GYRO-data & ACC-data & VLT-data & HALL-data                          
    x = np.zeros((numTests*config.numUB*config.numDS,config.numFeatures*(config.numSS-8)))                                                         # x:features, y:labels
    x[0:6*config.numUB*config.numDS, :] = allFeatures_N[0:6*config.numUB*config.numDS, config.numFeatures*7:config.numFeatures*9]
    x[6*config.numUB*config.numDS:, :] = allFeatures_N[8*config.numUB*config.numDS:9*config.numUB*config.numDS, config.numFeatures*7:config.numFeatures*9]
    '''
    
    '''
    ### 4.8 only with  CURT-data                          
    x = np.zeros((numTests*config.numUB*config.numDS,config.numFeatures*(config.numSS-9)))                                                         # x:features, y:labels
    x[0:6*config.numUB*config.numDS, :] = allFeatures_N[0:6*config.numUB*config.numDS, config.numFeatures*7:config.numFeatures*8]
    x[6*config.numUB*config.numDS:, :] = allFeatures_N[8*config.numUB*config.numDS:9*config.numUB*config.numDS, config.numFeatures*7:config.numFeatures*8]
    '''
    
    '''
    ### 4.9 only with  FON-data                         
    x = np.zeros((numTests*config.numUB*config.numDS,config.numFeatures*(config.numSS-9)))                                                         # x:features, y:labels
    x[0:6*config.numUB*config.numDS, :] = allFeatures_N[0:6*config.numUB*config.numDS, config.numFeatures*8:config.numFeatures*9]
    x[6*config.numUB*config.numDS:, :] = allFeatures_N[8*config.numUB*config.numDS:9*config.numUB*config.numDS, config.numFeatures*8:config.numFeatures*9]
    '''

    '''
    ### 4.10 only with HALL-data                        
    x = np.zeros((numTests*config.numUB*config.numDS,config.numFeatures*(config.numSS-9)))                                                         # x:features, y:labels
    x[0:6*config.numUB*config.numDS, :] = allFeatures_N[0:6*config.numUB*config.numDS,config.numFeatures*9:]
    x[6*config.numUB*config.numDS:, :] = allFeatures_N[8*config.numUB*config.numDS:9*config.numUB*config.numDS, config.numFeatures*9:]
    '''

    '''
    ### 4.11 with FON and HALL-data                        
    x = np.zeros((numTests*config.numUB*config.numDS,config.numFeatures*(config.numSS-8)))                                                         # x:features, y:labels
    x[0:6*config.numUB*config.numDS, :] = allFeatures_N[0:6*config.numUB*config.numDS,config.numFeatures*8:]
    x[6*config.numUB*config.numDS:, :] = allFeatures_N[8*config.numUB*config.numDS:9*config.numUB*config.numDS, config.numFeatures*8:]
    '''
      
   


    ###### 5. cross-validation
    skf = StratifiedKFold(n_splits=numTests)                           # seperate the samples in 7-Fold
    kfold = skf.split(x, y)
    
    ###### 6. kNN
    knn = KNeighborsClassifier(n_neighbors=4)
    k = 0
    for train_idx, test_idx in kfold:                                  # seperate the features of each fold for training und testing
        knn.fit(x[train_idx], y[train_idx])                            # training 
        prediction_knn = knn.predict(x[test_idx])                      # testing
        confusionM_knn = confusion_matrix(y[test_idx], prediction_knn) # confusion matrix
        accuracyS_knn = accuracy_score(y[test_idx, :], prediction_knn) # acceracy of the classification
        print('Fold ',k, ': ' 'accuracy score is: ', accuracyS_knn, 'confusion matrix is: ', confusionM_knn)
        k = k + 1
        
if __name__ == '__main__':
    mainScript()