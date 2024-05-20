'''
This script creates a class to store the functionality of a digital twin. This includes taking test data generated from '3 - Real World Simulation' for aging
under real world conditions. It makes probabilistic forecasts at a specified time for the health at the end of the simulation, assuming operating conditions do not change
and assesses this forecast accuracy with the continuous ranked probability score. It also makes deterministic predictions for the trajectory of health for a specified change
in operating conditions and assess this accuracy with the mean absolute percentage error score. 
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import simps

class DigitalTwin:
    '''
    This class forms the framework of a basic battery digital twin. When an object is created it takes in arguments for
    a data set to test its functionality. This data set contains health data over time where at a given point in time operating
    conditions are changed. The change in operating conditions are specified in an input dictionary. It also takes in an array of
    data frames containing lookup tables for the digital twin
    '''
    def __init__(self, data, operating_conditions, tables):

        #convert capacity to strictly decreasing soh data the data array contains  3 data frames, the fist fixed
        #operating conditions,the second has an increase in conditions at a given time and the third has a decrease at a given time.
        self.baseline_data = self.convert_data(data[0])
        self.increasing_data = self.convert_data(data[1])
        self.decreasing_data = self.convert_data(data[2])
        
        #save the data up until the operating conditions. For the test simulations this is 1520 hours
        self.start_data = self.starting_data(1520)

        #save the full array of time to fit models to
        self.time = np.arange(0,np.max(self.baseline_data[0]),10)

        #save the operating conditions that were used in the generation of the data
        self.operating_conditions = operating_conditions

        #save the lookup tables as an array of data frames and determine the bounds of parameter values
        self.tables = tables
        self.get_bounds()

        #save the weighting that is applied to the lookup table in the linear combination with the monte carlo parameters
        self.lookup_table_weighting = 0.75

    def convert_data(self, data_set):
        '''
        This helper function takes in a data frame containing capacity vs time and converts it to SOH against time while removing
        noisy erroneous results by removing data points that indicate an unexpected increase in SOH.

        INPUT: data frame containing capacity and time data
        OUTPUT: array of a time array and associated cleaned up SOH array for the given data set.
        '''
        #determine the state of health for the capacity measurement for the test data and extract the time points
        soh_c = np.array(data_set["Capacity"])/data_set["Capacity"][0]
        soh_t = np.array(data_set["Time"])

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

        return [soh_t,soh_c]
    
    def starting_data(self,change):
        '''
        This helper function extracts a portion of data corresponding to the time leading up to the change in operating conditions.

        INPUT: time at which operating conditions change
        OUTPUT: subset of the baseline data with only the extracted portion
        '''
        idx = np.abs(self.baseline_data[0] - change).argmin()
        starting_data = [self.baseline_data[0][:idx],self.baseline_data[1][:idx]]

        return starting_data

    def get_bounds(self):
        '''
        This helper function determines and save the bounds for the model parameters using the lookup tables
        '''

        #determine the bounds for the parameters using the max and min values from each table
        self.low_bound = []
        self.high_bound = []
        for table in self.tables:
            self.low_bound.append(table.drop(table.columns[[0, 1]], axis=1).min().min())
            self.high_bound.append(table.drop(table.columns[[0, 1]], axis=1).max().max())


    def degradation_model(self,x,a,b,c,d):
        '''
        This function defines the parametric empirical model that has been found to best describe the nature of battery degradation. It
        includes both a non-linear a linear portion of battery degradation

        INPUT: time data (x), model coefficients or parameters
        OUTPUT: the expected health of the battery with respect to the provided time 
        '''
        return a - b*x + np.exp(c / (x - d))
    
    def get_parameters(self, temp, soc):
        '''
        The helper function takes in specific values for SOC and temperature and fetches parameters values from the lookup tables saved as a class
        attribute.

        INPUT: Desired temp and soc value
        OUTPUT: The associated model parameters (A,B,C,D)
        '''

        #extract the available soc and temp values in the lookup table
        temp_vals = np.array(self.tables[0]["Temperature"])
        soc_vals = np.array(self.tables[0].columns[2:].astype(float).tolist())

        #find the closest row and column to the desired operating conditions
        temp_id = np.argmin(np.abs(temp_vals - temp))
        soc_col = str(soc_vals[np.argmin(np.abs(soc_vals - soc))])

        #create list to save the parameters
        params = []

        #loop through all the available lookup tables and get the parameter value
        for table in self.tables:
            val = table[soc_col][temp_id]
            params.append(val)

        return params
    
    def probabilistic_forecast(self):
        '''
        This function runs a monte carlo simulation on the baseline data (where operating conditions do not change)
        to generate a 1000 samples of predicted model parameters that are linearly combined with the lookup
        table parameters. These are saved as class attributes and the function does not return anything
        '''
        # define parameters for the monte carlo simulation
        samples = 500
        uncertainty = 0.05

        #extract the baseline operating conditions
        soc = self.operating_conditions["baseline"][0]
        temp = self.operating_conditions["baseline"][2]

        #extract the degradation model parameters from the lookup table
        parameters = self.get_parameters(temp, soc)
        lookup_A = parameters[0]
        lookup_B = parameters[1]
        lookup_C = parameters[2]
        lookup_D = parameters[3]

        #create arrays to save both degradation model and the linear benchmark parameters
        A = []
        B = []
        C = []
        D = []

        #for each available data points generate randomly sampled points for
        #monte carlo simulation
        for _ in range(samples):

            #create the randomly sampled points for the monte carlo simulation
            noise = np.random.normal(loc=0, scale=uncertainty**2, size=self.start_data[1].shape)
            samples = self.start_data[1] + noise

            p0 = list((np.array(self.low_bound) + np.array(self.high_bound))/2)

            #fit the degradation model
            popt = None
            try:
                popt, _ = curve_fit(self.degradation_model, self.start_data[0], samples, bounds = (self.low_bound, self.high_bound), p0=p0, maxfev=100000)
                a,b,c,d = popt
            except RuntimeError as e:
                a,b,c,d = popt

            #apply the weighted average with the lookup table parameters
            a = self.lookup_table_weighting*lookup_A + (1-self.lookup_table_weighting)*a
            b = self.lookup_table_weighting*lookup_B + (1-self.lookup_table_weighting)*b
            c = self.lookup_table_weighting*lookup_C + (1-self.lookup_table_weighting)*c
            d = self.lookup_table_weighting*lookup_D + (1-self.lookup_table_weighting)*d

            #save the parameters
            A.append(a)
            B.append(b)
            C.append(c)
            D.append(d)

        self.A_base, self.B_base, self.C_base, self.D_base = A, B, C, D

    def crps(self,vals,cdfs,true_value):
        '''
        This helper function calculates the normalized CRPS score for a probabilistic forecast given a known true value.

        INPUT: The sorted array of forecast samples and the associated cumulative distribution
        function of those samples and the true that the forecasts are aiming for.

        OUTPUT: The CRPS score of the probabilistic forecast
        '''
        
        # Calculate the Heaviside step function
        heaviside = np.where(vals >= true_value, 1, 0)
        
        # Calculate CRPS
        crps_value = simps((cdfs - heaviside) ** 2, vals)

        #calculate the square root of the crps as a percentage of the true value
        crps_value = (np.sqrt(crps_value)/true_value) *100
        
        return crps_value
    
    def cdf_baseline(self):
        '''
        This function makes a probabilistic forecast for the health of the battery at the end of the simulation, assuming that operating conditions do
        not change. It saves the values forecasted and the associated CDF and also saves the values predicted with 90% and 10% certainty to construct
        the 80% confidence interval. Everything is saved as class attributes and nothing is returned.
        '''

        #loop through all the samples of model parameters on the baseline data and save the forecast
        forecasts = []
        for i in range(len(self.A_base)):

            #fit the model with the randomly sampled model parameters
            sample = self.degradation_model(self.time,self.A_base[i],self.B_base[i],self.C_base[i],self.D_base[i])
            forecasts.append(sample[-1])

        #calculate the CDF of predictions
        self.cdf_baseline_val = np.sort(np.array(forecasts))
        self.cdf_baseline_cdf = np.cumsum(self.cdf_baseline_val) / np.sum(self.cdf_baseline_val)

        #determine the value at 90% and 10% confidence
        self.cdf_baseline_val_90 = self.cdf_baseline_val[np.argmin(np.abs(self.cdf_baseline_cdf - 0.9))]
        self.cdf_baseline_val_10 = self.cdf_baseline_val[np.argmin(np.abs(self.cdf_baseline_cdf - 0.1))]

    def get_baseline_cdf_vals(self):
        '''
        This helper function runs all the class methods for probabilistically forecasting health under no changes in operating conditions. It
        calculates the crps score for the probabilistic forecast and returns this with all the forecasted values.

        INPUT: None

        OUTPUT: values predicted by probabilistic forecast, CDF of predicted values,
        values predicted with 10% and 90% confidence and the CRPS score
        '''
        self.probabilistic_forecast()
        self.cdf_baseline()
        
        #calculate the crps score for the baseline forecast
        self.score = self.crps(self.cdf_baseline_val,self.cdf_baseline_cdf,self.baseline_data[1][-1])

        return self.cdf_baseline_val,self.cdf_baseline_cdf,self.cdf_baseline_val_90,self.cdf_baseline_val_10,self.baseline_data[1][-1],self.score

    def cdf_baseline_plot(self):
        '''
        Makes a plot of the cumulative distribution function for the forecasts of the health of the battery at the end of the simulation, assuming
        no changes to operating conditions. The forecasts are made before the change in operating condition. The function does not return anything.
        '''
        #define the min and max possible value at end
        x_min_pos = 0.97
        x_max_pos = 0.995
        y_min_pos = -0.1
        y_max_pos = 1.1

        #add in extremity points to improve plot legibility
        cdf_vals = np.concatenate(([x_min_pos], self.cdf_baseline_val, [x_max_pos]))
        cdf = np.concatenate(([0], self.cdf_baseline_cdf, [1]))

        #create array to plot true value
        true = self.baseline_data[1][-1]
        x = np.arange(x_min_pos,x_max_pos,0.0001)
        y = np.where(x < true, 0, 1)

        plt.plot(cdf_vals,cdf, color = 'blue', label = 'CDF of Forecast\nCRPS Score: ' + str(round(self.score,2)))

        #plot the confidence interval
        plt.plot([self.cdf_baseline_val_90, self.cdf_baseline_val_90], [y_min_pos, 0.9], color = 'black')
        plt.plot([x_min_pos, self.cdf_baseline_val_90], [0.9, 0.9], color = 'black')
        plt.plot([self.cdf_baseline_val_10, self.cdf_baseline_val_10], [y_min_pos, 0.1], color = 'black')
        plt.plot([x_min_pos, self.cdf_baseline_val_10], [0.1, 0.1], color = 'black', label = "80% Confidence\nInterval")

        plt.xlabel('Predicted SOH at ' + str(round(self.baseline_data[0][-1])) + " hours", fontsize = 14)
        plt.ylabel('Probability', fontsize = 14)
        plt.plot(x,y, color='red', label = 'True SOH at\n' + str(round(self.baseline_data[0][-1])) + " Hours")
        plt.legend(loc='center left', fontsize = 14)
        plt.xlim(x_min_pos, x_max_pos)
        plt.ylim(y_min_pos,y_max_pos)
        plt.show()
    
    def plot_probabilistic_baseline(self):
        '''
        This function makes a plot of monte carlo simulation trajectories using the degradation model to highlight the change in uncertainty over time.
        '''

        for i in range(len(self.A_base)):

            #fit the model with the randomly sampled model parameters
            sample = self.degradation_model(self.time,self.A_base[i],self.B_base[i],self.C_base[i],self.D_base[i])

            #plot the curve
            plt.plot(self.time,sample,linewidth = 0.15, color = 'red', alpha = 0.15)
        
        plt.plot(self.baseline_data[0][-1],self.baseline_data[1][-1], 'x', color = 'blue', label = 'True SOH at ' + str(round(self.baseline_data[0][-1])) + " hrs")
        plt.plot(self.start_data[0],self.start_data[1],'x', color = 'black', alpha = 1, label = 'Data Used to Make Forecast')
        plt.xlabel("Time /h", fontsize = 14)
        plt.ylabel("State of Health (SOH)", fontsize = 14)
        plt.legend(fontsize = 14)
        plt.ylim(0.98,1.005)
        plt.show()

    def what_if(self):
        '''
        This function uses the soh at the end of the initial data (leading up to the change in operating condition)
        It then produces 3 deterministic trajectories for the cases of:
        
        - operating conditions remain the same as the baseline
        - one or both of the operating conditions increase by a specified amount
        - one or both of the operating conditions decrease by a specified amount

        It makes a plot of the proposed trajectories and it returns the proportional change in linear degradation coefficient for each case.

        INPUT: None
        OUTPUT: The proportional change in linear degradation rate as a result of increasing or decreasing the operating conditions
        '''

        p0 = list((np.array(self.low_bound) + np.array(self.high_bound))/2)

        #first fit the degradation model to the first portion of the data to get the most representative SOH
        popt = None
        try:
            popt, _ = curve_fit(self.degradation_model, self.start_data[0], self.start_data[1], bounds = (self.low_bound, self.high_bound), p0=p0, maxfev=100000)
            a,b,c,d = popt
        except RuntimeError as e:
            a,b,c,d = popt

        soh = self.degradation_model(self.start_data[0][-1],a,b,c,d)

        #get the model parameters for three cases of operating conditions
        p = []
        A,B,C,D = self.get_parameters(self.operating_conditions["baseline"][2],self.operating_conditions["baseline"][0])
        p.append([A,B,C,D])
        A,B,C,D = self.get_parameters(self.operating_conditions["increasing"][2],self.operating_conditions["increasing"][0])
        p.append([A,B,C,D])
        A,B,C,D = self.get_parameters(self.operating_conditions["decreasing"][2],self.operating_conditions["decreasing"][0])
        p.append([A,B,C,D])

        #calculate the increase or decrease in linear degradation rate
        increase = p[1][1] / p[0][1]
        decrease = p[2][1] / p[0][1]

        #for each of the parameter sets, find the time at which SOH would equal the SOH at the time of the forecast
        x = np.arange(0,10000,10)
        self.what_ifs = []

        for i in p:
            model = self.degradation_model(x,i[0],i[1],i[2],i[3])
            t0 = x[np.abs(model - soh).argmin()]
            time = x + (self.start_data[0][-1] - t0)
            idx = np.where((time >= self.start_data[0][-1]) & (time <= self.baseline_data[0][-1]))[0]
            self.what_ifs.append([time[idx],model[idx]])

        #create labels for the plots.
        labels = ["No Change","15% Increase in SOC and Temp.","15% Decrease in SOC and Temp."]
        colors = ["black", "red", "blue"]

        for i in range(len(self.what_ifs)):
            plt.plot(self.what_ifs[i][0],self.what_ifs[i][1],label = labels[i], color = colors[i])

        plt.plot(self.start_data[0],self.start_data[1],'x',label = "Observed Health Measurements", color = 'black')
        plt.xlabel("Time /h" , fontsize = 14)
        plt.ylabel("State of Health (SOH)", fontsize = 14)
        plt.legend(fontsize = 14)
        plt.show()

        return increase,decrease

    def what_if_error(self):
        '''
        This helper function evaluates the normalized RMSE that considers the range of values that the SOH can realistically take by considering the bounds of the lookup table.
        
        It makes plots with the simulated data (after changing operating conditions)
        imposed on the what-if deterministic forecast. It returns the MAPE for both cases.

        INPUT: None
        OUTPUT: The mean absolute percentage error when making health trajectory predictions for both increasing and decreasing operating conditions
        '''

        #find the maximum possible SOH value that is expected given the length of the simulation
        #and the bounds of the lookup table
        max_soh = self.degradation_model(self.baseline_data[0][-1],self.high_bound[0],self.low_bound[1],self.high_bound[2],self.high_bound[3])
        min_soh = self.degradation_model(self.baseline_data[0][-1],self.low_bound[0],self.high_bound[1],self.low_bound[2],self.low_bound[3])
        soh_range = max_soh-min_soh

        #find the point at which the change in operating conditions starts
        start = np.abs(self.decreasing_data[0] - self.what_ifs[2][0][0]).argmin()

        #create arrays to save the absolute percentage error
        dec_se = []
        inc_se  = []

        #loop through each available data point, after operating conditions decrease, and save the absolute percentage error
        for data_point in range(len(self.decreasing_data[0][start:])):
            measured = self.decreasing_data[1][start:][data_point]
            predicted_index = np.abs(self.what_ifs[2][0] - self.decreasing_data[0][start:][data_point]).argmin()
            predicted = self.what_ifs[2][1][predicted_index]
            se = (measured - predicted) ** 2
            dec_se.append(se)

        # Repeat for the case of increasing operating conditions
        for data_point in range(len(self.increasing_data[0][start:])):
            measured = self.increasing_data[1][start:][data_point]
            predicted_index = np.abs(self.what_ifs[1][0] - self.increasing_data[0][start:][data_point]).argmin()
            predicted = self.what_ifs[1][1][predicted_index]
            se = (measured - predicted) ** 2
            inc_se.append(se)

        #calculate NRMSE for both cases
        dec_nrmse = np.sqrt(np.mean(np.array(dec_se)))/soh_range
        inc_nrmse = np.sqrt(np.mean(np.array(inc_se)))/soh_range

        plt.plot(self.decreasing_data[0], self.decreasing_data[1], 'x', label="Simulated Data", color="blue")
        plt.plot(self.what_ifs[2][0], self.what_ifs[2][1], label="Prediction Made with Digital Twin,\nNRMSE = " + str(round(dec_nrmse, 3)), color="blue")
        plt.plot(self.start_data[0], self.start_data[1], 'x', color="black")
        plt.xlabel("Time /h", fontsize = 14)
        plt.ylabel("State of Health (SOH)", fontsize = 14)
        plt.legend(fontsize = 14)
        plt.show()

        plt.plot(self.increasing_data[0], self.increasing_data[1], 'x', label="Simulated Data", color="red")
        plt.plot(self.what_ifs[1][0], self.what_ifs[1][1], label="Prediction Made with Digital Twin,\nNRMSE = " + str(round(inc_nrmse, 3)), color="red")
        plt.plot(self.start_data[0], self.start_data[1], 'x', color="black")
        plt.xlabel("Time /h", fontsize = 14)
        plt.ylabel("State of Health (SOH)", fontsize = 14)
        plt.legend(fontsize = 14)
        plt.show()

        return dec_nrmse,inc_nrmse
