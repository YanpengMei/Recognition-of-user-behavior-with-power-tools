# This script is to combine the features of all sensor-datas from Test008 to Test021 and save it in a csv-file
import numpy as np
import pandas as pd
import os
from configuration import configuration as config

###### 1. get all features
path =  os.path.dirname(os.getcwd()) + "/data/Test"    
testsName = ['008','009','010','011','012','013','015','016','018','019','020','021']             # data in test014 and test017 are not used
numTests = 12                                                                                     # number of tests                                                                     # first test is test008
features_all = np.zeros((config.numDS*config.numUB*numTests, config.numFeatures*config.numSS))    # empty array for all features, see also configuration.py
for i in range(numTests):                                                                         # fill the empty array with values
    path4features = path + testsName[i] + '/1_features' + testsName[i] + '.csv'
    features = np.array(pd.read_csv(path4features, header=None))
    features[np.isnan(features)] = 0
    if i != (numTests-1):
        features_all[config.numDS*config.numUB*i:config.numDS*config.numUB*(i+1),:] = features
    else:
        features_all[config.numDS*config.numUB*i:,:] = features

###### 2. save features in csv    
path2SaveFeatures = path + '021bis008/allFeatures.csv'                                             # path to save the data
features2Save = pd.DataFrame(features_all)                                                         # to be saved variable
features2Save.to_csv(path2SaveFeatures, header=False, index=False)                                 # save variable to a .csv-dokument

###### 3. check
print(features_all.shape)                                                                          # should be (1440,310)