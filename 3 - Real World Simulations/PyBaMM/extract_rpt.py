'''
This script creates a helper function to take a full data set of rpt tests and split it out into individual csv files, each containing a single pulse or
a single full discharge. These are used to determine the SOH of the battery at a given time. This function is directly called within this script can be be
adjusted depending on the file path of the data that needs to be extracted.
'''

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

def extract_rpt(path):
    '''
    This function takes in the file location of a csv file data and splits out each individual RPT test into a separate csv file to be analyzed separately.

    INPUT: file path (relative to working directory) of the folder containing the simulation RPT tests.
    OUTPUT: None
    '''
    
    #load in the full data of all RPT tests
    data = pd.read_csv(path + "/full.csv")
    
    #extract the voltage from the data to detect where each RPT tests starts and stops
    voltage = np.array(data["Terminal Voltage"])

    #find the end points for the rpt tests
    idx1 = np.where((voltage > 4.19) & (np.abs(np.insert(np.diff(voltage), 0, 0)) < 0.01))[0]
    idx1 = idx1[np.where(np.insert(np.diff(idx1), 0, 0)>100)]
    idx1 = np.insert(idx1, 0, 0)

    idx2, _ = find_peaks(-voltage, prominence=0.005, height=-2.55)
    idx2 = np.append(idx2,len(voltage)-1)
    idx2 = np.insert(idx2[np.where(np.insert(np.diff(idx2), 0, 0)>20)],0,idx2[0])

    idx = np.sort(np.concatenate([idx1,idx2]))

    #if incomplete set of data remove data points so multiple of 3
    if len(idx) % 3 != 0:
        n = len(idx) % 3
        idx = idx[:-n]

    flag = 0
    windows = []

    #convert the spikes into windows to extract data
    for i in idx:
        if flag == 0:
            start1 = i
            flag+=1
        elif flag == 1:
            end1 = i
            start2 = i
            flag+=1
        elif flag == 2:
            end2 = i
            windows.append([[start1,end1],[start2,end2]])
            flag = 0 
    
    #loop through the windows and save the data in csv files
    for i in range(len(windows)):

        #define the file paths for the discharge and the GITT pulses
        dis_path = path + "/discharge-" + str(i) + ".csv"
        gitt_path = path + "/gitt-" + str(i) + ".csv"

        #split the input data
        dis = data[windows[i][0][0]: windows[i][0][1]]
        gitt = data[windows[i][1][0]: windows[i][1][1]]

        #export the data as csv files
        dis.to_csv(dis_path, index=False)
        gitt.to_csv(gitt_path, index=False)

#extract the RPTs for a given simulation
extract_rpt("Data/08-05-55")
extract_rpt("Data/09-05-25")



