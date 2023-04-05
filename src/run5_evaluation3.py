# This script is to compare different machine-learning algorithmus. Only features from CURT-data will be used
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold
from configuration import configuration as config


def mainScript():
    ###### 1. get path and allFeatures
    path = os.path.dirname(os.getcwd()) + '/data/Test021bis008'     # main path
    path4features = path + '/allFeatures.csv'                       # path for allFeatures.csv
    allFeatures = np.array(pd.read_csv(path4features, header=None)) # read all features

    ###### 2. normalization of allFeatures
    normalize = StandardScaler()
    allFeatures_N = normalize.fit_transform(allFeatures)            # normalized features

    ###### 3. get features from CURT
    numTests = 7                                                    # data in test 8,9,10,11,12,13,18 will be used
    x = np.zeros((numTests*config.numUB*config.numDS, config.numFeatures*(config.numSS-9))) # x:features, y:labels
    x[0:6*config.numUB*config.numDS,:] = allFeatures_N[0:6*config.numUB*config.numDS, config.numFeatures*7:config.numFeatures*8]
    x[6*config.numUB*config.numDS:,:] = allFeatures_N[8*config.numUB*config.numDS:9*config.numUB*config.numDS, config.numFeatures*7:config.numFeatures*8]

    ###### 4. add labels
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

    ###### 5 cross-validation
    skf = StratifiedKFold(n_splits=numTests)                           # spperate the samples in 7-fold
    kfold = skf.split(x, y)




    ###### 6. Evaluation 3: compare different algorithmus
    '''
    ### 6.1 kNN
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=4)
    k = 0
    for train_idx, test_idx in kfold:                                  # seperate the features of each fold for training und testing
        knn.fit(x[train_idx], y[train_idx])                            # training
        prediction_knn = knn.predict(x[test_idx])                      # testing
        confusionM_knn = confusion_matrix(y[test_idx], prediction_knn) # confusion matrix
        accuracyS_knn = accuracy_score(y[test_idx, :], prediction_knn) # acceracy of the classification
        print('Fold ',k, ': ' 'accuracy score is: ', accuracyS_knn, 'confusion matrix is: ', confusionM_knn)
        k = k + 1
    '''

    '''
    ### 6.2 svm
    from sklearn import svm
    svm_model = svm.SVC(kernel='linear', C=1, gamma=1)                # 'C=1': normalization
    k = 0
    for train_idx, test_idx in kfold:
        svm_model.fit(x[train_idx], y[train_idx])
        prediction_svm = svm_model.predict(x[test_idx])
        confusionM_svm = confusion_matrix(y[test_idx], prediction_svm)
        accuracyS_svm = accuracy_score(y[test_idx, :], prediction_svm)
        print('Fold ',k, ': ' 'accuracy score is: ', accuracyS_svm, 'confusion matrix is: ', confusionM_svm)
        k = k + 1
    '''

    '''
    ### 6.3 Decision Tree
    from sklearn.tree import DecisionTreeClassifier
    dtc = DecisionTreeClassifier()
    k = 0
    for train_idx, test_idx in kfold:
        dtc.fit(x[train_idx], y[train_idx])
        prediction_dtc = dtc.predict(x[test_idx])
        confusionM_dtc = confusion_matrix(y[test_idx], prediction_dtc)
        accuracyS_dtc = accuracy_score(y[test_idx, :], prediction_dtc)
        print('Fold ',k, ': ' 'accuracy score is: ', accuracyS_dtc, 'confusion matrix is: ', confusionM_dtc)
        k = k + 1
    '''

    '''
    ### 6.4 Random Forest
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier()
    k = 0
    for train_idx, test_idx in kfold:
        rfc.fit(x[train_idx], y[train_idx])
        prediction_rfc = rfc.predict(x[test_idx])
        confusionM_rfc = confusion_matrix(y[test_idx], prediction_rfc)
        accuracyS_rfc = accuracy_score(y[test_idx, :], prediction_rfc)
        print('Fold ',k, ': ' 'accuracy score is: ', accuracyS_rfc, 'confusion matrix is: ', confusionM_rfc)
        k = k + 1
    '''

    #'''
    ### 6.5 Gradient Boosting
    from sklearn.ensemble import GradientBoostingClassifier
    gbc = GradientBoostingClassifier()
    k = 0
    for train_idx, test_idx in kfold:
        gbc.fit(x[train_idx], y[train_idx])
        prediction_gbc = gbc.predict(x[test_idx])
        confusionM_gbc = confusion_matrix(y[test_idx], prediction_gbc)
        accuracyS_gbc = accuracy_score(y[test_idx, :], prediction_gbc)
        print('Fold ',k, ': ' 'accuracy score is: ', accuracyS_gbc, 'confusion matrix is: ', confusionM_gbc)
        k = k + 1
    #'''
        



if __name__ == '__main__':
    mainScript()