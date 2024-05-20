'''
This script
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from GPR import run_GPR

def degradation_model(x,a,b,c,d):
    '''
    This function defines the parametric empirical model that has been found to best describe the nature of battery degradation. It
    includes both a non-linear a linear portion of battery degradation

    INPUT: time data (x), model coefficients or parameters
    OUTPUT: the expected health of the battery with respect to the provided time 
    '''
    return a - b*x + np.exp(c / (x - d))

def arrhenius_rate(T):
    '''
    This function defines the Arrhenius rate law and includes constants from the simulation parameters (Okane2022)

    INPUT: temperature values (T) or array from 15 to 55
    OUTPUT: array of Arrhenius rate constant at the associated temperatures
    '''
    #convert Temperature to Kelvin
    T = T + 273.15

    # Gas constant in J/mol·K
    R = 8.314

    # Pre-exponential factor in s^-1
    A = 1.0e13

    # Activation energy in J/mol from OKane2022 parameter set
    Ea = 38000

    return A * np.exp(-Ea / (R * T))

def extract_relationship(paths):
    '''
    This helper function takes an array of paths to data frames that either simulate degradation at a fixed SOC and varying temperature, or
    at a fixed temperature and varying SOC. It returns an array of linear degradation coefficient, from the degradation model, for each data frame.

    INPUT: array of data frames with health data over time for a varying operating conditions.
    OUTPUT: array of associated linear degradation coefficients.
    '''

    #create an array to save the linear degradation coefficients
    linear_rate = []

    for path in paths:
        df = pd.read_csv(path)

        #determine the state of health for the capacity measurement and extract the time points
        soh = np.array(df["Capacity"])/df["Capacity"][0]
        time = np.array(df["Time"])

        #fit the degradation model to the available samples to extract the linear degradation rate
        try:
            popt, _ = curve_fit(degradation_model, time, soh, p0=[0, 0.001, 10, -6], maxfev=1000)
            _, B, _, _ = popt
        except RuntimeError as e:
            _, B, _, _ = popt

        linear_rate.append(B)
    
    return np.array(linear_rate)

#specify what soc values are of interest at a fixed temperature of 25 degrees
socs = [0.4,0.6,0.7,0.8,1.0]
soc_range = np.arange(socs[0]-0.2,socs[-1],0.01)

#create file paths to all the necessary data frames
soc_data_paths = []
for soc in socs:
    soc_data_paths.append("Data/Varying SOC/" + f'{soc:.2f}'.replace('.', '')[:-1] + "-25.csv")

#extract the linear degradation rate with respect to SOC
linear_rate_soc = extract_relationship(soc_data_paths)

# model the relationship between linear degradation rate and soc with GPR
soc_model,_ = run_GPR(socs,linear_rate_soc,soc_range,length=0.8,sigma=5, noise = 0.005)

# load in the anode half-cell discharge voltage curve to impose on the plot
half_cell_data = pd.read_csv("Data/half-cell-voltage.csv")
half_cell_soc = half_cell_data["SOC"]
half_cell_voltage = half_cell_data["Anode Half-Cell Discharge Voltage"]

# plot plot from 20% SOC upwards
idx = np.where(half_cell_soc > 0.2)[0]
half_cell_soc = half_cell_soc[idx]
half_cell_voltage = half_cell_voltage[idx]

#plot the linear degradation rate with respect to SOC
fig, ax1 = plt.subplots(figsize=(8, 6))
line1, = ax1.plot(soc_range, soc_model * 100, color="blue", label="Modelled SOH Degradation Rate")
ax1.set_xlabel("State of Charge (SOC)", fontsize=14)
ax1.set_ylabel("Modelled SOH Degradation Rate / % per hour", fontsize=14)

ax2 = ax1.twinx()
line2, = ax2.plot(half_cell_soc, half_cell_voltage, '--', color='red', label='Anode Half-Cell Discharge Voltage')
ax2.set_ylabel('Half-Cell Voltage /V', fontsize=14)
ax2.set_ylim(0.1,0.7)

lines = [line1, line2]
labels = [line.get_label() for line in lines]
plt.legend(lines, labels, fontsize=14)
plt.show()

#specify what temperature values are of interest at a fixed soc of 40%
temps = [15,25,35,55]
temp_range = np.arange(temps[0]-15,temps[-1]+15,0.5)

#create file paths to all the necessary data frames
temp_data_paths = []
for temp in temps:
    temp_data_paths.append("Data/Varying Temp/04-" + str(temp) + ".csv")

# extract the linear degradation rate with respect to temperature
linear_rate_temp = extract_relationship(temp_data_paths)

# model the relationship between linear degradation rate and temperature with GPR
temp_model,_ = run_GPR(temps,linear_rate_temp,temp_range,length=50,sigma=10, noise = 0.005)

#plot the Arrhenius rate
rate = arrhenius_rate(temp_range)

# Plot the linear degradation rate with respect to Temperature
fig, ax1 = plt.subplots(figsize=(8, 6))
line1, = ax1.plot(temp_range, temp_model * 100, color="blue", label="Modelled SOH Degradation Rate")
ax1.set_xlabel("Temperature /°C", fontsize=14)
ax1.set_ylabel("Modelled SOH Degradation Rate / % per hour", fontsize=14)

ax2 = ax1.twinx()
line2, = ax2.plot(temp_range, rate, '--', color='red', label='Ahrrenius Rate Constant')
ax2.set_ylabel('Ahrrenius Rate Constant', fontsize=14)
ax2.set_ylim(0,1.4e7)

lines = [line1, line2]
labels = [line.get_label() for line in lines]
plt.legend(lines, labels, fontsize=14)
plt.show()


