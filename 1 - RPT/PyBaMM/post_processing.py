'''
This script takes the data from the solution of the simulation (run in prt_simulation.py), splits out individual pulses into separate csv files to be processes
independently and performs coulomb counting to determine OCV with respect to SOC.

Note - for this script to run, the working directory must be 'Battery Digital Twin/1 - RPT'
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema
from scipy.signal import find_peaks

def filter_unique(data, threshold=0.00001):
    '''
    This helper function remove flat points from the find peaks by keeping only values that are unique

    INPUT: array of data
    OUTPUT: array with only unique values
    '''

    #create empty array to save the result and sort the input data
    result = []
    sorted_data = sorted(data)

    #loop through the input data and save values that are not within a threshold of another value
    for i in range(len(sorted_data)):
        current_value = sorted_data[i]
        if i == 0 or abs(current_value - sorted_data[i - 1]) > threshold:
            if i == len(sorted_data) - 1 or abs(current_value - sorted_data[i + 1]) > threshold:
                result.append(current_value)

    return result

def filter_similar(data, threshold = 200):
    '''
    This helper function removes rogue peaks by removing peaks that are within a given range from another peak

    INPUT: array of data
    OUTPUT: array with only unique values
    '''

    #prepare input data
    data = data.tolist() if isinstance(data, np.ndarray) else data
    i = 0

    #keep only data that is unique
    while i < len(data) - 1:
        current_value = data[i]
        next_value = data[i + 1]
        if abs(next_value - current_value) <= threshold:
            data.pop(i)
        else:
            i += 1

    return data

def voltage_peaks(terminal_voltage):
    '''
    This helper function takes an array of voltage values and returns the indexes of where peak
    values occur

    INPUT: Voltage array
    OUTPUT: array of indexes corresponding with peak voltages
    '''

    #find troughs of current pulses and filter out flat sections with helper functions
    minima_indices = argrelextrema(terminal_voltage, np.less)[0]
    filtered_data = filter_unique(terminal_voltage[minima_indices[1:]])
    time_indexes = np.where(np.isin(terminal_voltage, np.array(filtered_data)))[0]

    #find the trough after the full discharges
    discharge_trough, _ = find_peaks(-np.array(terminal_voltage), height=-3)

    #combine the peaks and filter out peaks that are close together
    time_indexes = np.concatenate((time_indexes, discharge_trough))
    time_indexes = np.insert(time_indexes, 0, minima_indices[0])
    time_indexes = np.sort(time_indexes)
    time_indexes = filter_similar(time_indexes)

    return time_indexes

def get_windows(time,ids):
    '''
    This helper function takes the array of times and an array of indexes for the current pulses
    return list of start and end times to fit each pulse

    INPUT: time array, indexes for times of interest.
    OUTPUTS: array of arrays of start and end times corresponding with the input indexes
    '''
    windows = []
    for i in ids:
        windows.append([time[i]-0.025,time[i]+0.175])

    return(windows)

def coulomb_counting(data, startah):
    '''
    This helper function takes an array of time and current data and a starting capacity and performs
    coulomb counting to estimate the soc at a given time.

    INPUT: data frame with time and current data, the capacity of the cell when the data starts
    OUTPUT: An array of the available discharge capacity with respect to time
    '''

    #extract data from input data frame, assume time interval is constant
    current = np.array(data["Current"])
    time = np.array(data["Time"])
    time_interval_seconds = round((time[1]-time[0])*60*60,2)

    #calculate discharge capacity in As and Ah using cumulative of charge transferred
    charge = current*-time_interval_seconds
    startas = startah * 60*60
    charge = np.insert(charge,0,startas)
    capacity = np.cumsum(charge)
    capacity_ah = capacity/(60*60)

    #remove the last term to account for starting value and return
    capacity_ah = capacity_ah[:-1]
    return capacity_ah

def get_soc(data):
    '''
    This helper function calculates the soc at any given time given a data frame containing the estimate
    discharge capacity against time (obtained using coulomb counting)

    INPUT: data frame containing estimate discharge capacity against time.
    OUTPUT: an array wit the corresponding SOC
    '''
    #extract data from input data frame
    capacity = np.array(data["Estimate Discharge Capacity"])
    max_cap = np.max(capacity)

    #calculate soc as a ratio of the discharge capacity to initial capacity
    soc = capacity/max_cap

    return soc

def closest_value(array, value):
    '''
    This helper function finds the index of the value in an array that is closest to a given value.

    INPUT: data array and value of interest
    OUTPUT: the index of the value in the array that is closest to the input value
    '''
    closest_index = min(range(len(array)), key=lambda i: abs(array[i] - value))
    return closest_index

#function that takes in a profile, determines the SOC at any given time and splits it into pulses
#it also returns two lists with SOC and corresponding OCV
def process_profile(profile, temp):
    '''
    This function takes in data from a simulation, determines the SOC at any given time and splits it into pulses
    It makes an estimate of the cell OCV against time using the SOC and saves each pulse as an independent csv file.

    INPUT: The full RPT test simulation output profile at a given temperature, the temperature of interest.
    OUTPUT: None
    '''

    #use the temperature to determine where the files should be stored
    directory = "data/" + str(temp)

    #estimate the discharge capacity and SOC for the whole profile
    profile["Estimate Discharge Capacity"] = coulomb_counting(profile, 0)
    profile["Estimate SOC"] = get_soc(profile)
    profile["Estimate SOC"] = round(profile["Estimate SOC"],3)

    #save the voltage and time data and find the indexes of the peaks
    terminal_voltage = np.array(profile["Terminal Voltage"])
    time = np.array(profile["Time"])

    #find the time indexes for relevant information
    #the first two are data points are for the slow discharge, the rest are for the current pulses
    time_indexes = voltage_peaks(terminal_voltage)
    discharge_window = [time[time_indexes[0]],time[time_indexes[1]]]
    windows = get_windows(time,time_indexes[3:len(time_indexes)])

    # extract the voltage profile for the steady discharge section in the RPT test
    discharge = profile[profile["Time"].between(discharge_window[0], discharge_window[1])]

    #fit the ocv curve using using the slow discharge phase
    lookup_soc = sorted(list(set(discharge["Estimate SOC"])), reverse = True)
    lookup_ocv = []
    lookup_capacity = []

    for soc in lookup_soc:
        indexes = [i for i, x in enumerate(discharge["Estimate SOC"]) if x == soc]
        ocv = np.mean(np.array(discharge["Terminal Voltage"].iloc[indexes]))
        capacity = np.mean(np.array(discharge["Estimate Discharge Capacity"].iloc[indexes]))
        lookup_ocv.append(ocv)
        lookup_capacity.append(capacity)

    #create a data frame to save the OCV curve
    ocv_curve = pd.DataFrame({"SOC":lookup_soc, "Discharge Capacity": lookup_capacity, "OCV": lookup_ocv})
    
    #export the ocv curve as a csv
    filepath = directory + "/" + str(temp) + "-ocv-curve.csv"
    ocv_curve.to_csv(filepath, index=False)

    #loop through all the sampling windows for the current pulses
    for i in range(len(windows)):

        #extract the data
        pulse = profile[profile["Time"].between(windows[i][0], windows[i][1])]

        #find the starting ocv and therefore the starting capacity
        initial_ocv = pulse["Terminal Voltage"].iloc[0]
        initial_index = closest_value(lookup_ocv,initial_ocv)
        initial_capacity = lookup_capacity[initial_index]

        #Perform Coloumb counting from the starting capacity
        capacity = coulomb_counting(pulse, initial_capacity)

        #use the capacity to find the ocv profile across the pulse
        ocv = []
        for cap in capacity:
            ocv.append(lookup_ocv[closest_value(lookup_capacity,cap)])

        #append the ocv
        pulse["Estimate OCV"] = ocv

        #export csv
        pulse.to_csv(directory + "/Pulses/" + str(i+1) + ".csv", index=False)

#define what temperatures are used
temp_c = np.array([10,20,30,40])
temp_k = temp_c + 273.15

#loop through all temperatures and split all the data after determining the estimate OCV against time
for temp in temp_k:

    # read in the full rpt test data for the given temperature
    profile = pd.read_csv("Data/" + str(temp) + "/" + str(temp) + "-rpt-test.csv")

    #process the profile by splitting it into individual pulses after determining the ocv against time
    process_profile(profile, temp)