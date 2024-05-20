'''
This script calls in data for degradation under specific operating conditions, from **'Data'** and fits three models to the data,
namely a linear model, a square root time model and an empirical model considering the the non-linear and linear phase of degradation,
otherwise known as the 'degradation model'. Each fit is plotted and R scores are compared
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def degradation_model(x,a,b,c,d):
    '''
    This function defines the parametric empirical model that has been found to best describe the nature of battery degradation. It
    includes both a non-linear a linear portion of battery degradation

    INPUT: time data (x), model coefficients or parameters
    OUTPUT: the expected health of the battery with respect to the provided time 
    '''
    return a - b*x + np.exp(c / (x - d))

def square_root_time_model(x,a,b):
    '''
    This function defines the square root time model used as a function to model degradation with respect to time.

    INPUT: time data (x), model coefficients or parameters
    OUTPUT: the expected health of the battery with respect to the provided time 
    '''
    return a* np.power(x,(1/2)) + b

def linear_model(x,a,b):
    '''
    This function defines the linear y=mx+c model used as a function to model degradation with respect to time, as a benchmark

    INPUT: time data (x), model coefficients or parameters
    OUTPUT: the expected health of the battery with respect to the provided time 
    '''
    return a*x + b

#load in the degradation data for specific operating conditions
df = pd.read_csv("Data/04-55/extracted.csv")

#determine the state of health for the capacity measurement and extract the time points
soh = np.array(df["Capacity"])/df["Capacity"][0]
time = np.array(df["Time"])

#create an array of time values to visualise the curve
x = np.arange(time[0],time[-1],1)

#fit the degradation model to the data points
popt = None
try:
    popt, pcov = curve_fit(degradation_model, time, soh, p0=[0, 0.001, 10, -6], maxfev=1000)
    A, B, C, D = popt
except RuntimeError as e:
    A, B, C, D = popt

#plot the degradation model fit
y = degradation_model(x,A, B, C, D)

#calculate the R score
y_pred = degradation_model(time, A, B, C, D)
r_squared = r2_score(soh, y_pred)

plt.plot(time,soh,'x',color = "black", label = "Data Points from Simulation")
plt.plot(x,y, color = "red", label = "Degradation Model Approximation\nR Score = " + str(round(r_squared,3)))
plt.xlabel("Time /h", fontsize = 14)
plt.ylabel("State of Health (SOH)", fontsize = 14)
plt.legend(fontsize = 14)
plt.show()

#fit the square root time model to the data points
popt = None
try:
    popt, pcov = curve_fit(square_root_time_model, time, soh, p0=[-0.005, 1], maxfev=1000)
    A, B = popt
except RuntimeError as e:
    A, B = popt

#plot the degradation model fit
y = square_root_time_model(x, A, B)

#calculate the R score
y_pred = square_root_time_model(time, A, B)
r_squared = r2_score(soh, y_pred)

plt.plot(time,soh,'x',color = "black", label = "Data Points from Simulation")
plt.plot(x,y, color = "red", label = "Square Root Time Model Approximation\nR Score = " + str(round(r_squared,3)))
plt.xlabel("Time /h", fontsize = 14)
plt.ylabel("State of Health (SOH)", fontsize = 14)
plt.legend(fontsize = 14)
plt.show()

#fit the linear model to the data points
popt = None
try:
    popt, pcov = curve_fit(linear_model, time, soh, p0=[-0.005, 1], maxfev=1000)
    A, B = popt
except RuntimeError as e:
    A, B = popt

#plot the degradation model fit
y = linear_model(x, A, B)

#calculate the R score
y_pred = linear_model(time, A, B)
r_squared = r2_score(soh, y_pred)

plt.plot(time,soh,'x',color = "black", label = "Data Points from Simulation")
plt.plot(x,y, color = "red", label = "Linear Model Approximation\nR Score = " + str(round(r_squared,3)))
plt.xlabel("Time /h", fontsize = 14)
plt.ylabel("State of Health (SOH)", fontsize = 14)
plt.legend(fontsize = 14)
plt.show()