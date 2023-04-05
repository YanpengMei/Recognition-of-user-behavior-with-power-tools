# This script is to see the influence of different labeling-situations. Only features of CURT-data will be used
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
    path = os.path.dirname(os.getcwd()) + '/data/Test021bis008'      # main path
    path4features = path + '/allFeatures.csv'                        # path for allFeatures.csv
    allFeatures = np.array(pd.read_csv(path4features, header=None))  # read features

    ###### 2. normalization of allFeatures
    normalize = StandardScaler()
    allFeatures_N = normalize.fit_transform(allFeatures)             # normalized features

    ###### 3. get features from CURT-data
    numTests = 7                                                     # data in test 8,9,10,11,12,13,18 will be used
    x = np.zeros((numTests*config.numUB*config.numDS, config.numFeatures*(config.numSS-9))) # x:features, y:labels
    x[0:6*config.numUB*config.numDS,:] = allFeatures_N[0:6*config.numUB*config.numDS, config.numFeatures*7:config.numFeatures*8]
    x[6*config.numUB*config.numDS:,:] = allFeatures_N[8*config.numUB*config.numDS:9*config.numUB*config.numDS, config.numFeatures*7:config.numFeatures*8]
    



    ###### 4. Evaluation 2: compare two ways of adding labels
    '''
    ### 4.1 add labels - situation 1
    # meaning of labels:
    #                   1: screw in, screws A, wood 
    #                   2: screw out, screws A, wood
    #                   3: screw in, screws B, wood 
    #                   4: screw out, screws B, wood
    #                   5: drill, wood
    #                   6: drill, Alu
    y = np.zeros((numTests*config.numUB*config.numDS,1))
    for j in range(numTests):
        for i in range(config.numUB): 
            if i == (config.numUB-1) and j == (numTests - 1):
                y[(j*config.numUB*config.numDS + i*config.numDS):] = i+1
            else:
                y[(j*config.numUB*config.numDS + i*config.numDS):(j*config.numUB*config.numDS + (i+1)*config.numDS)] = i+1
    '''

    '''
    ### 4.2 add labels situation 2
    # meaning of labels:
    #                   1: screw in
    #                   2: screw out
    #                   3: drill
    y = np.zeros((numTests*config.numUB*config.numDS,1))
    for j in range(numTests):
        for i in range(config.numUB): 
            if i == (config.numUB-1) and j == (numTests - 1):
                y[(j*config.numUB*config.numDS + i*config.numDS):] = 3
            else:
                if i==0 or i==2:
                    y[(j*config.numUB*config.numDS + i*config.numDS):(j*config.numUB*config.numDS + (i+1)*config.numDS)] = 1
                if i==1 or i==3:
                    y[(j*config.numUB*config.numDS + i*config.numDS):(j*config.numUB*config.numDS + (i+1)*config.numDS)] = 2
                if i==4 or i==5:
                    y[(j*config.numUB*config.numDS + i*config.numDS):(j*config.numUB*config.numDS + (i+1)*config.numDS)] = 3
    '''

    '''
    ### 4.3 add labels situation 3
    # meaning of labels:
    #                   1: screws A
    #                   2: screws B
    #                   3: 7.5 mm twist drill bits
    y = np.zeros((numTests*config.numUB*config.numDS,1))
    for j in range(numTests):
        for i in range(config.numUB): 
            if i == (config.numUB-1) and j == (numTests - 1):
                y[(j*config.numUB*config.numDS + i*config.numDS):] = 3
            else:
                if i==0 or i==1:
                    y[(j*config.numUB*config.numDS + i*config.numDS):(j*config.numUB*config.numDS + (i+1)*config.numDS)] = 1
                if i==2 or i==3:
                    y[(j*config.numUB*config.numDS + i*config.numDS):(j*config.numUB*config.numDS + (i+1)*config.numDS)] = 2
                if i==4 or i==5:
                    y[(j*config.numUB*config.numDS + i*config.numDS):(j*config.numUB*config.numDS + (i+1)*config.numDS)] = 3
    '''
    
    #'''
    ### 4.4 add labels with situation 4
    # meaning of labels:
    #                   1: wood 
    #                   2: alu.
    y = np.zeros((numTests*config.numUB*config.numDS,1))
    for j in range(numTests):
        for i in range(config.numUB): 
            if i == (config.numUB-1) and j == (numTests - 1):
                y[(j*config.numUB*config.numDS + i*config.numDS):] = 2
            else:
                if i>=0 and i<=4:
                    y[(j*config.numUB*config.numDS + i*config.numDS):(j*config.numUB*config.numDS + (i+1)*config.numDS)] = 1
                if i==5:
                    y[(j*config.numUB*config.numDS + i*config.numDS):(j*config.numUB*config.numDS + (i+1)*config.numDS)] = 2
    #'''

    '''
    #plot to check if labels are rightly labeled
    temp = np.linspace(1,numTests*config.numUB*config.numDS,numTests*config.numUB*config.numDS)
    plt.figure()
    plt.plot(temp,y,'b')
    plt.show()
    '''




    ###### 5. cross-validation                                         
    skf = StratifiedKFold(n_splits=numTests)                            # spperate the samples in 7-fold
    kfold = skf.split(x, y)
    
    ###### 6. kNN
    knn = KNeighborsClassifier(n_neighbors=4)
    k = 0
    for train_idx, test_idx in kfold:                                   # seperate the features of each fold for training und testing
        knn.fit(x[train_idx], y[train_idx])                             # training
        prediction_knn = knn.predict(x[test_idx])                       # testing
        confusionM_knn = confusion_matrix(y[test_idx], prediction_knn)  # confusion matrix
        accuracyS_knn = accuracy_score(y[test_idx, :], prediction_knn)  # acceracy of the classification
        print('Fold ',k, ': ' 'accuracy score is: ', accuracyS_knn, 'confusion matrix is: ', confusionM_knn)
        k = k + 1
        
if __name__ == '__main__':
    mainScript()