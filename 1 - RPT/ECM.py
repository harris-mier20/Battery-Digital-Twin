'''
This script provides helper functions to first make an estimate for 1RC ECM parameters given a current pulse and voltage profile. It then uses Scipy
tools to fit the ECM parameters with Nelder-Mead numerical optimization by minimizing the square difference between the expected terminal voltage and
ECM output. These helper functions are used in lookup_table.py.
'''
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from scipy.signal import argrelextrema

def initial_estimate(df):
    '''
    This function provides estimates for R0, R1, C1 given a data frame containing time, current and terminal voltage
    for a current pulse

    INPUT: data frame with time, current and voltage data
    OUTPUT: dictionary containing estimates for R0, R1 and C1 ECM parameters.
    '''

    # Extract the data - these are the column names needed from the imported data
    voltage = np.array(df["Terminal Voltage"])
    current = max(df["Current"])
    time = df["Time"]

    #caluculate the standard deviation of the voltage data to help with identifying peaks
    std_voltage = np.std(voltage)

    # Find the base of the voltage response and its voltage value
    trough = argrelextrema(voltage, np.less)[0]
    trough = trough[trough >= 5]
    trough_voltage = voltage[int(trough)]

    #search for a voltage change more than 1 standard deviation away
    for i in range(int(trough), len(voltage)):
        if (voltage[i] > (trough_voltage + std_voltage)):
            voltage_jump = i
            break

    #use the standard deviation to determine when the voltage has settled.
    voltage_steady = len(voltage)-10
    prev_voltage = voltage_jump
    for i in range((voltage_jump+1), len(voltage)):
        if (voltage[i] < (voltage[prev_voltage] + (0.0005*std_voltage))):
            voltage_steady = i
            break
        else :
            prev_voltage = i

    #calculate R0. R0 is the voltage jump divided by the current
    R0 = (voltage[voltage_jump]-trough_voltage)/current

    #use this value of R0 to find R1. the steady state voltage difference
    R1 = ((voltage[voltage_steady]-trough_voltage)-(current*R0))/current

    #now we find C1 assuming the time to converge is 4*R1*C1
    C1 = ((time[voltage_steady]-time[int(trough)])*60*60) / (4*R1)

    #find the interval between data points
    dt = round((time[1]-time[0])*60*60)

    #prepare the output dictionary
    output = {
        "R0":R0,
        "R1":R1,
        "C1":C1,
        "dt":dt,
    }

    return output

#create function that takes in a data frame for a pulse and return the
#1RC parameter values
def fit_ECM(df):
    '''
    This function provides numerically optimized values of R0, R1 and C1 after making an initial estimate.

    INPUT: data frame with time, current and voltage data
    OUTPUT: dictionary containing optimized R0, R1 and C1 ECM parameters.
    '''

    #determine estimates of the RC parameters with the helper function
    estimate = initial_estimate(df)
    r0_0 = estimate["R0"]
    r1_0 = estimate["R1"]
    c1_0 = estimate["C1"]
    dt = estimate["dt"]

    #get data for OCV, expected voltage and known current
    ocv = df['Estimate OCV']
    expected_voltage = df['Terminal Voltage']
    current = df["Current"]

    # nested function for determining the voltage response
    def fit_ECM(r0,r1,c1):
        '''
        This function is nested in the optimization function. It determines the voltage output
        of the 1RC ECM model given RC parameters.

        INPUT: RC parameters
        OUTPUT: Associated voltage profile of ECM
        '''

        #create list to store the calculated voltage, determine required length
        ecm_voltage = []
        data_length = len(ocv)
        ir = 0

        #loop through current data
        for i in range(data_length):

            #apply the equation for the 1RC ECM
            ir = ir * math.exp(-dt / (r1*c1)) + (current[i] * (1 - (math.exp(-dt / (r1*c1)))))
            vr = r1 * ir
            v_new = ocv[i] - r0*current[i] - vr

            ecm_voltage.append(v_new)

        return ecm_voltage

    # nested function to calculate the rmse for given RC parameters
    def rmse(p):
        model = fit_ECM(p[0],p[1],p[2])
        mse = mean_squared_error(expected_voltage, model)
        rmse = np.sqrt(mse)
        return rmse

    # determine the initial RC estimates and bounds
    initial_variables = [r0_0,r1_0,c1_0]
    bounds = [(0.0000001, 1), (0.0000001, 1), (100, 50000)]

    # use the nelder mead method to find RC parameters that minimize the difference between expected and modelled voltage
    result = minimize(rmse, initial_variables, bounds=bounds, method='Nelder-Mead')
    optimized_variables = result.x

    output = {
        "R0": optimized_variables[0],
        "R1": optimized_variables[1],
        "C1": optimized_variables[2],
        "RMSE": result.fun
    }

    return output