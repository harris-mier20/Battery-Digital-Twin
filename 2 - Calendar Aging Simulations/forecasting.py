'''
This script implements an empirical model that was found to represent the nature of degradation well. It includes both a linear and
non-linear factor. This model is fit to a subset of data points with a simulated measurement uncertainty to make a forecast for the
remainder of the data points. This functionality is developed further in '4 - Digital Twin Framework'.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

def degradation_model(x,a,b,c,d):
    '''
    This function defines the parametric empirical model that has been found to best describe the nature of battery degradation. It
    includes both a non-linear a linear portion of battery degradation

    INPUT: time data (x), model coefficients or parameters
    OUTPUT: the expected health of the battery with respect to the provided time 
    '''
    return a - b*x + np.exp(c / (x - d))

#define the number of data points available at the time of the forecast
n_points = 8

#import the health data
df = pd.read_csv('data/08-25/extracted.csv')

#determine the state of health for the capacity measurement and extract the time points
soh = np.array(df["Capacity"])/df["Capacity"][0]
time = np.array(df["Time"])

#extract the time at which the battery reaches a health of 98%
end_health = np.interp(0.98, np.flip(soh), np.flip(time))

#extract the time at which the forecast is issued given only a select few data points are available
issue_time = time[n_points-1]

#create an array of time to fit the model to
full_time = np.arange(time[0],time[-1]+2000,5)

#create array to store predictions for the time taken to reach 98.25% SOH
predictions = []

#create a plot
fig, axs = plt.subplots(2, 1, figsize=(8, 8))

#run 500 monte carlo simulations with the empirical degradation model 
for i in range(500):

    # generate gaussian noise to simulate measurement uncertainty
    uncertainty = np.random.normal(loc=0, scale=0.006 * (np.max(soh[:n_points]) - np.min(soh[:n_points])), size=soh[:n_points].shape)

    # Add noise to the data to create the samples
    samples = soh[:n_points] + uncertainty

    #fit the degradation model to the available samples to make a forecast for the time to reach 98% SOH
    try:
        popt, pcov = curve_fit(degradation_model, time[:n_points], samples, p0=[0, 0.001, 10, -6], maxfev=1000)
        A, B, C, D = popt
    except RuntimeError as e:
        A, B, C, D = popt

    trajectory = degradation_model(full_time,A,B,C,D)

    #extract the time at which the trajectory crosses 98% SOH
    id = np.where(trajectory < 0.98)[0][0]
    predictions.append(full_time[id])
    
    #plot the trajectory
    axs[0].plot(full_time, trajectory, color = 'red' , linewidth = 0.5, alpha=0.1)

# add the sub set and full set of data points to the plot
axs[0].plot(time, soh, 'x', color = 'black', alpha = 0.5, label = "Full Simulation Data")
axs[0].plot(time[:n_points], soh[:n_points], 'x', color = 'black', label = "Data Points Available At Forecast")
axs[0].plot(full_time, np.ones(len(full_time))*(0.98), '--', color = 'black')
axs[0].set_xlabel("Time /h", fontsize = 14)
axs[0].set_ylabel("State of Health (SOH)", fontsize = 14)
axs[0].legend(fontsize = 14)

#create probability distribution plot for the time to reach 98% SOH
kde_plot = sns.kdeplot(predictions, shade=False, color='blue', ax=axs[1], label = "Predicted Time to Reach 98% SOH")

max_density = np.max(kde_plot.get_lines()[0].get_data()[1])
peak_index = np.argmax(kde_plot.get_lines()[0].get_data()[1])
peak_value = kde_plot.get_lines()[0].get_data()[0][peak_index]

axs[1].set_xlabel('End of Life /h', fontsize = 14)
axs[1].set_ylabel('Probability Density', fontsize = 14)
axs[1].axvline(x=end_health, color='red', linestyle='--', label = "Measured Time to Reach 98% SOH")
axs[1].legend(fontsize = 14)

plt.tight_layout()
plt.show()
         


