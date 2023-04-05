'''
This script is aimed to calculate the features of each measured data
Abbreviations of sensors: GYRO_X,GYRO_Y,GYRO_Z,ACC_X,ACC_Y,ACC_Z,VLT,CURT,HALL(for hall sensor),FON(for mikrofon)
Order of the 6 sensors will always be:  GYRO_X,GYRO_Y,GYRO_Z,ACC_X,ACC_Y,ACC_Z,VLT,CURT,HALL,FON
Numbers, that are representing sensors:GYRO_X,GYRO_Y,GYRO_Z,ACC_X,ACC_Y,ACC_Z,VLT,CURT,HALL,FON --> 1,2,3,4,5,6,7,8,9,10
'''
from multiprocessing import Pool
import numpy as np
from configuration import configuration as config
from scipy import stats
import pywt
from scipy.fftpack import fft
import math
import matplotlib.pyplot as plt


class getFeatures:
    def getF_main(dataSeg):
        ###### 1. multi-processing with 6 processors
        inputList = [(dataSeg[0],1),(dataSeg[1],2),(dataSeg[2],3),(dataSeg[3],4),(dataSeg[4],5), (dataSeg[5],6),(dataSeg[6],7),(dataSeg[7],8),(dataSeg[8],9),(dataSeg[9],10)]
        pool = Pool(6)                                                          # crete a pool with 6 processors
        output = pool.map(getFeatures.calFeatures, inputList)
        pool.close()
        pool.join()

        ###### 2. arrange the order of features
        numF = config.numFeatures                                               # number of features of all sensordatas
        featureMatrix = np.zeros((config.numDS*config.numUB,config.numSS*numF)) # see configuration.py
        for i in range(config.numSS):
            if output[i][1] == i+1 and i != 9:
                featureMatrix[:, i*numF:(i+1)*numF] = output[i][0]
            if output[i][1] == 10 and i == 9:
                featureMatrix[:, i*numF:] = output[i][0]
        return featureMatrix




    def calFeatures(inputTuple):
        '''This is the function to calculate the features from the inputData
        name, meaning and order of the features are:
        f_duration      duration of one behavior
        f_min:          minimum value
        f_max:          maximum value
        f_mean:         mean value
        f_median:       median value
        f_SD:           standard deviation
        f_RMS:          root mean square
        f_IR:           interquartile range
        f_P1:           percentiles 15%
        f_P2:           percentiles 30%
        f_P3:           percentiles 45%
        f_P4:           percentiles 60%
        f_P5:           percentiles 75%
        f_P6:           percentiles 90%
        f_S:            skewness
        f_K:            kurtosis
        f_ZCR:          zero crossing rate
        f_MCR:          mean crossing rate
        f_RMS_DW1_cA:   root mean square of daubechies wavelet 1 cA
        f_RMS_DW1_cD:   root mean square of daubechies wavelet 1 cD
        f_RMS_DW2_cA:   root mean square of daubechies wavelet 2 cA
        f_RMS_DW2_cD:   root mean square of daubechies wavelet 2 cD
        f_RMS_DW3_cA:   root mean square of daubechies wavelet 3 cA
        f_RMS_DW3_cD:   root mean square of daubechies wavelet 3 cD
        f_RMS_DW4_cA:   root mean square of daubechies wavelet 4 cA
        f_RMS_DW4_cD:   root mean square of daubechies wavelet 4 cD
        f_FFT_meanF:    mean frequence
        f_FFT_median:   median frequence
        f_HPAS1:        1st highest peak in amplitude spectrum
        f_HPAS2:        2nd highest peak in amplitude spectrum
        f_HPAS3:        3rd highest peak in amplitude spectrum
        '''
        features = np.zeros((config.numDS*config.numUB, config.numFeatures))    # number of features is in configuration.py
        for i in range(config.numDS*config.numUB):
            DS_i = inputTuple[0][i]
            ###### 1. features in time domain
            features[i, 0] = DS_i.shape[0]/config.SF                            # duration
            features[i, 1] = np.min(np.absolute(DS_i))                          # min
            features[i, 2] = np.max(np.absolute(DS_i))                          # max
            features[i, 3] = np.mean(np.absolute(DS_i))                         # mean
            features[i, 4] = np.median(np.absolute(DS_i))                       # median
            features[i, 5] = np.std(DS_i)                                       # standard deviation
            features[i, 6] = getFeatures.getRMS(DS_i)                           # root mean square
            features[i, 7] = np.percentile(DS_i, 75) - np.percentile(DS_i, 25)  # interquartile range
            features[i, 8] = np.percentile(DS_i, 15)                            # percentiles 15%
            features[i, 9] = np.percentile(DS_i, 30)                            # percentiles 30%
            features[i, 10] = np.percentile(DS_i, 45)                           # percentiles 45%
            features[i, 11] = np.percentile(DS_i, 60)                           # percentiles 60%
            features[i, 12] = np.percentile(DS_i, 75)                           # percentiles 75%
            features[i, 13] = np.percentile(DS_i, 90)                           # percentiles 90%
            features[i, 14] = stats.skew(DS_i)                                  # skewness
            features[i, 15] = stats.kurtosis(DS_i)                              # kurtosis
            features[i, 16] = getFeatures.ZCR(DS_i)                             # zero crossing rate
            features[i, 17] = getFeatures.MCR(DS_i)                             # mean crossing rate
            ###### 2. features from daubechies wavelet
            features[i, 18], features[i, 19], features[i, 20], features[i, 21], features[i, 22], features[i, 23], features[i, 24], features[i, 25] = getFeatures.myDW(DS_i)
            ###### 3. features from FFT
            features[i, 26], features[i, 27], features[i, 28], features[i, 29], features[i, 30] = getFeatures.myFFT(DS_i)
        outputTuple = (features, inputTuple[1])
        return outputTuple




    def myDW(data):
        ###### 1. get root mean square of daubechies wavelet 1-4
        cA1, cD1 = pywt.dwt(data.T.tolist(), 'db1')
        cA2, cD2 = pywt.dwt(data.T.tolist(), 'db2')
        cA3, cD3 = pywt.dwt(data.T.tolist(), 'db3')
        cA4, cD4 = pywt.dwt(data.T.tolist(), 'db4')
        RMS_DW1_cA = getFeatures.getRMS(cA1)
        RMS_DW1_cD = getFeatures.getRMS(cD1)
        RMS_DW2_cA = getFeatures.getRMS(cA2)
        RMS_DW2_cD = getFeatures.getRMS(cD2)
        RMS_DW3_cA = getFeatures.getRMS(cA3)
        RMS_DW3_cD = getFeatures.getRMS(cD3)
        RMS_DW4_cA = getFeatures.getRMS(cA4)
        RMS_DW4_cD = getFeatures.getRMS(cD4)
        ###### 2. plot and print to analyze
        # plt.figure()
        # plt.plot(data.T.tolist()[0],'r')
        # plt.plot(pywt.idwt(cA1, cD1, 'db1'),'b')
        # plt.show()
        # print(len(data.T.tolist()[0]), cA1.shape, cD1.shape)
        # print(RMS_DW1_cA, RMS_DW1_cD, RMS_DW2_cA, RMS_DW2_cD, RMS_DW3_cA, RMS_DW3_cD, RMS_DW4_cA, RMS_DW4_cD)
        return RMS_DW1_cA, RMS_DW1_cD, RMS_DW2_cA, RMS_DW2_cD, RMS_DW3_cA, RMS_DW3_cD, RMS_DW4_cA, RMS_DW4_cD




    def myFFT(data):
        # get meanF, medianF, HPAS1, HPAS2, HPAS3, see also "calFeatures"
        ###### 1. do fft
        myfft = fft(data)
        fft_amp_normalized = np.abs(myfft) / data.shape[0]                            # normalized amplitude
        fft_amp_normalized_half = fft_amp_normalized[range(int(data.shape[0]/2))]     # take leftside half
        ###### 2. find peaks in fft_amp_normalized
        peaks = getFeatures.findPeaks(fft_amp_normalized_half)                        # all peaks 
        ###### 3. get output
        meanF, medianF, HPAS1, HPAS2, HPAS3 = (None, None, None, None, None)
        if len(peaks) >= 1:                                                           # highest peak
            meanF = np.mean(peaks)                                                    # mean value of peaks
            medianF = np.median(peaks)                                                # median value of peaks
            HPAS1 = np.max(fft_amp_normalized_half[peaks])                            # highest peak
        if len(peaks) >= 2:                                                           
            HPAS2 = np.sort(fft_amp_normalized_half[peaks], axis=0)[-2]               # second highest peak
        if len(peaks) >= 3:                                                           
            HPAS3 = np.sort(fft_amp_normalized_half[peaks], axis=0)[-3]               # third highest peak
        ###### 4. plot and print to analyze
        # f = np.linspace(0, fft_amp_normalized_half.shape[0] - 1, fft_amp_normalized_half.shape[0])
        # if dataName == 8:
        #    plt.figure()
        #    plt.plot(f, fft_amp_normalized_half)
        #    plt.plot(peaks, fft_amp_normalized_half[peaks],'o')
        #    plt.show()
        #    print(meanF, medianF, HPAS1, HPAS2, HPAS3,'\n')
        return meanF, medianF, HPAS1, HPAS2, HPAS3




    def findPeaks(data):
        ''' this function is aimed to find peaks in order 2'''
        ###### 1. first order
        peaks = []
        if data[0] > data[1]:
            peaks = peaks + [0]
        for i in range(1, data.shape[0] - 1):
            if (data[i] >= data[i - 1] and data[i] > data[i + 1]) or (
                    data[i] > data[i - 1] and data[i] >= data[i + 1]):
                peaks = peaks + [i]
        if data[-1] > data[-2]:
            peaks = peaks + [data.shape[0] - 1]
        ###### 2. second order
        if len(peaks) >= 4:
            peaks_2 = []
            if data[peaks[0]] > data[peaks[1]]:
                peaks_2 = peaks_2 + [peaks[0]]
            for i in range(1, len(peaks) - 1):
                if (data[peaks[i]] >= data[peaks[i - 1]] and data[peaks[i]] > data[peaks[i + 1]]) or (
                        data[peaks[i]] > data[peaks[i - 1]] and data[peaks[i]] >= data[peaks[i + 1]]):
                    peaks_2 = peaks_2 + [peaks[i]]
            if data[peaks[-1]] > data[peaks[-2]]:
                peaks_2 = peaks_2 + [peaks[-1]]
        else:
            peaks_2 = peaks
        ###### 3. use a filter
        remember = []                                          # remember which element to delete
        for i in range(len(peaks_2) - 1):
            if data[peaks_2[i + 1]] == data[peaks_2[i]]:
                remember = remember + [i]
        if len(remember) > 0:
            for i in reversed(remember):
                del peaks_2[i]
        return peaks_2




    def getRMS(data):
        # get root mean square
        RMS = math.sqrt(sum(data ** 2) / data.shape[0])
        return RMS



    def ZCR(data):
        num_ZC = 0                                               # number of zero crossing
        for i in range(data.shape[0] - 1):
            if (data[i + 1] > 0 and data[i] < 0) or (data[i + 1] < 0 and data[i] > 0):
                num_ZC = num_ZC + 1
            ZCR = 100 * num_ZC / data.shape[0]
        return ZCR



    def MCR(data):
        num_MC = 0                                               # number of mean crossing
        meanVal = np.mean(data)
        for i in range(data.shape[0] - 1):
            if (data[i + 1] > meanVal and data[i] < meanVal) or (data[i + 1] < meanVal and data[i] > meanVal):
                num_MC = num_MC + 1
            MCR = 100 * num_MC / data.shape[0]
        return MCR
