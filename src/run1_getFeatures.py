# This script is the main script to get features of each sensor datas
import os
import pandas as pd
from classes.preProcessing import preProcessing as prePro
from classes.getFeatures import getFeatures as getF


def mainScript():
    testName = '009'                                                          # test number, from '008' to '021'
    mainPath = os.path.dirname(os.getcwd()) + "/data\Test" + testName         # path of test-folder
    allSensorData = prePro.prePro_main(mainPath)                              # get all sensor data
    allSensorData_seg = prePro.segmentation(mainPath, allSensorData)          # extract the data according to the CURT-Value into segments
    featureMatrix = getF.getF_main(allSensorData_seg)                         # calculate the features for each data segment
    
    # save features in csv    
    path2SaveFeatures = mainPath + '/1_features' + testName + '.csv'          # path to save the data
    features2Save = pd.DataFrame(featureMatrix)                               # to be saved variable
    features2Save.to_csv(path2SaveFeatures, header=False, index=False)        # save variable to a .csv-dokument


if __name__ == '__main__':
    mainScript()