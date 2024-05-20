'''
This script takes in the current and time data from 'ion-o-raw-data.csv' and smooths and discretizes it before reformatting it as a
series of current steps and associated durations. It exports the reformatted data as a csv saved as 'drive_cycle_steps.csv'.
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def convert_str(val):
    '''
    This helper function converts a time value as as string in form hh:mm:ss to seconds as an integer.

    INPUT: time string in format hh:mm:ss
    OUTPUT: integer value of time in seconds
    '''
    hours, minutes, seconds = map(int, val.split(':'))
    total_seconds = hours * 3600 + minutes * 60 + seconds
    
    return total_seconds

#vectorise the conversion function with numpy
convert_str = np.vectorize(convert_str)

def exponential_smoothing(series, alpha):
    '''
    This helper function to used to perform exponential smoothing on data.

    INPUT: array of noisy data and a float value for the alpha level of the exponential smoothing.
    OUTPUT: array of exponentially smoothed data
    '''
    smoothed_series = np.zeros_like(series, dtype=float)
    smoothed_series[0] = series[0]

    #loop through all values in the series and apply exponential smoothing
    for i in range(1, len(series)):
        smoothed_series[i] = alpha * series[i] + (1 - alpha) * smoothed_series[i - 1]

    return smoothed_series

def discretize(data, increments):
    '''
    This helper function is used to discretize an array of data. It fits every data point into discrete bins of given increments
    and return the new array

    INPUT: continuous data array, increments of bin values to discretize the array
    OUTPUT: The discretized array
    '''
    discretized_data = []

    #loop through all the data
    for i in data:

        #round to the nearest increment
        x = round(i / increments) * increments
        discretized_data.append(x)

    return np.array(discretized_data)

def c_duration(current,time):
    '''
    This helper function converts a discretized current array and time array and reformats it to an array of current steps and another array
    the associated durations for each step. This can be interpreted by PyBaMM.

    INPUT: discretized current profile and associated time array
    OUTPUT: tuple of current steps and associated durations for each step
    '''
    #create variables to save results
    current_steps = []
    duration_steps = []
    c = 0
    t = 0

    #loop through all the current data points and determine the time since
    #the current last changed
    for i in range(len(current)):
        if (current[i] != c):
            current_steps.append(c)
            c = current[i]
            duration_steps.append(time[i]-t)
            t = time[i]

    return np.array(current_steps),np.array(duration_steps)

#import the raw data
raw = pd.read_csv("Data/ion-o-raw-data.csv")

#save the date time as a string
string_time = np.array(raw["DateTime"])

#convert it to numeric values
time = convert_str(string_time)

#extract current and voltage and save it as a float
current = np.array(raw["Current"]).astype(float)
voltage = np.array(raw["Voltage"]).astype(float)

#define the start and end time of the single journey to extract
start = 48300
end = 50400
idx = np.where((time < end) & (time > start))[0]

#cut out the relevant data for the single profile
time = time[idx]
voltage = voltage[idx]
current = -current[idx]

#reset the time
time = time - np.min(time)

# perform exponential smoothing on the current profile
current = exponential_smoothing(current,0.05)

#discretize the current profile to fit current at any given time into bins of 2.2 A increments
current = discretize(current,1)

#normalise the data so that the mean is 1- can later multiply by a constant to select
#the average discharge current for the profile
current = current/np.mean(current)

#convert to current and duration
c_steps,d_steps = c_duration(current,time)

#remove steps where duration is 0
idx = np.where(d_steps != 0)[0]
c_steps = c_steps[idx]
d_steps = d_steps[idx]

#export the profile as current and duration
df = pd.DataFrame()
df["Current"]=c_steps
df["Duration"]=d_steps
df.to_csv("Data/drive_cycle_steps.csv")