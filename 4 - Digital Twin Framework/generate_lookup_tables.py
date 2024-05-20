'''
This script uses the simulation outputs at sampled operating conditions to determine the degradation model
coefficients at these operating conditions before fitting GPR curves in 2 dimensions (variations of each coefficient
with respect to both SOC and Temperature). For each coefficient, it creates a 2D lookup table which is exported as a
csv and saved in 'Data'.
'''

import pandas as pd
import numpy as np
from GPR import run_GPR
from scipy.optimize import curve_fit
import os

def degradation_model(x,a,b,c,d):
    '''
    This function defines the parametric empirical model that has been found to best describe the nature of battery degradation. It
    includes both a non-linear a linear portion of battery degradation

    INPUT: time data (x), model coefficients or parameters
    OUTPUT: the expected health of the battery with respect to the provided time 
    '''
    return a - b*x + np.exp(c / (x - d))

def get_parameters(file_name):
    '''
    This helper function takes the name of a file (referring to the extracted output of a simulation at specific operating conditions).
    It then determines the SOH with respect to time and fits the degradation model to the data points. It then returns the degradation model
    coefficients as an array as well as the operating conditions that were used in the associated simulation together in a tuple.

    INPUT: name, as a string, of a csv containing health data of a simulation.
    OUTPUT: The degradation model coefficients as an array and the associated operating conditions
    '''

    #Load in the data
    data = pd.read_csv("Data/0.5A Simulations/" + file_name)

    #determine the state of health for the capacity measurement and extract the time points
    soh = np.array(data["Capacity"])/data["Capacity"][0]
    soh_c = soh
    soh_t = np.array(data["Time"])

    #create arrays to find only points where the soh is strictly decreasing 
    dec_soh_c = []
    dec_soh_t = []

    #loop through all available data points and check if the soh is decreasing
    prev = 10
    for i in range(len(soh_c)):
        if soh_c[i] < prev:
            dec_soh_c.append(soh_c[i])
            dec_soh_t.append(soh_t[i])
            prev = soh_c[i]

    #save the new data points
    soh_c = np.array(dec_soh_c)
    soh_t = np.array(dec_soh_t)

    #fit the exponential curve to the new data points
    popt = None
    try:
        popt, _ = curve_fit(degradation_model, soh_t, soh_c, p0=[0, 0.001, 10, -6], maxfev=1000)
        A, B, C, D = popt
    except RuntimeError as e:
        A, B, C, D = popt

    #find the average soc, current and temperature from the file name
    conditions = [int(i) for i in file_name.rstrip('.csv').split('-')]

    #reformat the soc so it is in the format 0.XX
    conditions[0] = conditions[0] / (10 ** len(str(conditions[0])))

    #reformat the average discharge current so it is in the format 0.XX
    conditions[1] = conditions[1] / (10 ** len(str(conditions[1])))

    #reformat the temperature so it is in the format XX.XX
    conditions[2] = conditions[2] / (10 ** (len(str(conditions[2]))-2))

    return [A,B,C,D], conditions

#create empty array to store the model coefficients from the sampled operating conditions
gt_data = []

#get all the available data files from simulations
files = os.listdir("Data/0.5A Simulations")

#loop through all the available data and save the parameters that were fit
for file in files:
    parameters,conditions = get_parameters(file)
    gt_data.append([conditions,parameters])

#create an array to save the available socs for which a GPR will be fit to
socs = []

#pull out the available socs and sort them in an array
for i in gt_data:
    if i[0][0] not in socs:
        socs.append(i[0][0])

socs = sorted(socs)

#create data frame to save the fit GPR data at each soc
A_df = pd.DataFrame()
B_df = pd.DataFrame()
C_df = pd.DataFrame()
D_df = pd.DataFrame()

#define the range of temperatures and socs to consider for the lookup table
temp_range = np.arange(10,55,1)
soc_range = np.arange(0.6,0.8,0.02)

#for each soc get a list of parameters for the available temperatures
for soc in socs:

    #create empty list to store temps that are being used and the parameter values
    #for the given soc
    temps = []
    As = []
    Bs = []
    Cs = []
    Ds = []

    #get all the parameters at respective temperatures
    for i in gt_data:
        if i[0][0] == soc:
            temps.append(i[0][2])
            As.append(i[1][0])
            Bs.append(i[1][1])
            Cs.append(i[1][2])
            Ds.append(i[1][3])

    #fit GPR to the parameters at different temperatures
    A_range,_ = run_GPR(temps,As,temp_range,length=30,sigma=0.1)
    B_range,_ = run_GPR(temps,Bs,temp_range,length=30,sigma=0.1)
    C_range,_ = run_GPR(temps,Cs,temp_range,length=30,sigma=0.1)
    D_range,_ = run_GPR(temps,Ds,temp_range,length=30,sigma=0.1)

    A_df[soc] = A_range
    B_df[soc] = B_range
    C_df[soc] = C_range
    D_df[soc] = D_range

#create new data frames for the final lookup tables for each parameter
A_lookup = pd.DataFrame(columns = soc_range)
B_lookup = pd.DataFrame(columns = soc_range)
C_lookup = pd.DataFrame(columns = soc_range)
D_lookup = pd.DataFrame(columns = soc_range)

#remove the erroneous second data point
socs = np.concatenate((np.array(socs)[:1], np.array(socs)[2:]))

#now loop through every row of the data frame and fit a GPR to the soc range of interest
for row in range(len(A_df)):
    
    #get values of the associated row from each data frame
    A_vals = A_df.iloc[row].values
    B_vals = B_df.iloc[row].values
    C_vals = C_df.iloc[row].values
    D_vals = D_df.iloc[row].values

    #remove the erroneous second data point
    A_vals = np.concatenate((np.array(A_vals)[:1], np.array(A_vals)[2:]))
    B_vals = np.concatenate((np.array(B_vals)[:1], np.array(B_vals)[2:]))
    C_vals = np.concatenate((np.array(C_vals)[:1], np.array(C_vals)[2:]))
    D_vals = np.concatenate((np.array(D_vals)[:1], np.array(D_vals)[2:]))

    #fit a GPR to each of the rows
    A_range,_ = run_GPR(socs,A_vals,soc_range,length=0.8,sigma=5)
    B_range,_ = run_GPR(socs,B_vals,soc_range,length=0.8,sigma=5)
    C_range,_ = run_GPR(socs,C_vals,soc_range,length=0.8,sigma=5)
    D_range,_ = run_GPR(socs,D_vals,soc_range,length=0.8,sigma=5)

    A_lookup = pd.concat([A_lookup, pd.DataFrame([list(A_range)], columns=A_lookup.columns)], ignore_index=True)
    B_lookup = pd.concat([B_lookup, pd.DataFrame([list(B_range)], columns=B_lookup.columns)], ignore_index=True)
    C_lookup = pd.concat([C_lookup, pd.DataFrame([list(C_range)], columns=C_lookup.columns)], ignore_index=True)
    D_lookup = pd.concat([D_lookup, pd.DataFrame([list(D_range)], columns=D_lookup.columns)], ignore_index=True)

#add the temperature to the first column
A_lookup.insert(0, "Temperature", temp_range)
B_lookup.insert(0, "Temperature", temp_range)
C_lookup.insert(0, "Temperature", temp_range)
D_lookup.insert(0, "Temperature", temp_range)

#export all the lookup tables
A_lookup.to_csv("Data/A-lookup.csv")
B_lookup.to_csv("Data/B-lookup.csv")
C_lookup.to_csv("Data/C-lookup.csv")
D_lookup.to_csv("Data/D-lookup.csv")