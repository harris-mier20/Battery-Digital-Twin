'''
This script creates a class that uses the coloumb counting and equivalent circuit modelling developed in '1 - RPT' to determine the SOH of the battery at a given time
using data frames containing time,current and voltage data for a full discharge and a current pulse
'''

import numpy as np
from scipy.signal import find_peaks
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import math
import pandas as pd

class RPT:
    '''
    class to process the RPT data and calculate and save the RC parameters and capacity at the time that the
    test was run. It is initiated with data frames containing time,current and voltage data for a full discharge
    and a current pulse.
    '''
    def __init__(self,Discharge,GITT):

        #store the data frames for the GITT pulses and the full discharge
        self.Discharge = Discharge
        self.GITT = GITT

        #run the parameterize method to parameterize the ECM model and
        #determine the max discharge capacity at the time of the RPT
        self.pulse1 = self.parameterize()

        #save the total available capacity
        self.capacity = self.pulse1['Max Capacity']

        #save the internal resistance as determined by the current pulse
        self.ecm_resistance = self.pulse1["R0"]+self.pulse1["R1"]

    def parameterize(self):
        '''
        Takes the data frame for the GITT pulses and slow discharge from an RPT and returns the
        capacity and RC parameters at that given time. Uses several nested functions to process the data
        '''

        def pulse_intervals(voltage):
            '''
            This helper function takes in a voltage profile determines the indexes for the start and end of each GITT pulse

            INPUT: array of voltage data
            OUTPUT: array of indexes for the start and end of each GITT pulse
            '''

            #calculate discrete derivative
            diff = np.diff(voltage)
            diff = np.insert(diff, 0, diff[0])

            #find the peaks of the voltages
            ppeak, _ = find_peaks(diff)
            npeak, _ = find_peaks(-diff)
            peaks = np.concatenate((ppeak, npeak))
            peaks = np.sort(peaks)

            #filter out the peaks we need
            peaks = peaks[np.abs(diff[peaks]) >= 0.005]
            peaks = peaks[(peaks >= len(voltage)/3) & (peaks <= 2*len(voltage)/3)]

            peaks[0] -=8
            peaks[1] +=60  

            return peaks

        def coulomb_counting(current,time):
            '''
            This helper function takes a current and time profile and performs coulomb counting to determine the
            capacity at any given time in Ah

            INPUT: current and time data arrays for a full discharge
            OUTPUT: the total available capacity of the cell
            '''
            #calculate the charge transferred in Ah
            time_diff = np.diff(time)
            time_diff = np.insert(time_diff, 0, time_diff[0])
            charge = time_diff * current

            #calculate the cumulative charge transferred
            cumulative = np.cumsum(charge)

            #determine the maximum capacity and therefore the soc at any other time
            cumulative = abs(np.min(cumulative)) + cumulative

            return cumulative

        def estimate_resistance(voltage,current):
            '''
            This helper function makes an estimate for the steady state internal resistance of the cell (R0 + R1). It takes the
            voltage and current profile of a pulse and uses dV = dI*(R0+R1), where dV is the difference in voltage between the
            minimum and steady state voltage. This will be used to slightly correct the terminal voltage during the discharge to appear closer to the
            true OCV

            INPUT: voltage and current data for a GITT pulse
            OUTPUT: The estimate steady state internal resistance of the cell
            '''

            #find the index in the voltage array that corresponds with the minimum voltage during the pulse
            index = np.argmin(voltage)

            #use the index to find the current in mA and the difference in voltage
            min_voltage = voltage[index]
            steady_voltage = voltage[-1]
            dV = steady_voltage-min_voltage
            dI = current[index]

            #use the defined values to calculate and return the estimate internal resistance
            resistance = dV/dI
            return resistance

        def fit_param_values (ocv,voltage,current,time,init_r0,init_r1,init_c=15000):
            '''
            This helper function uses estimates for ECM parameter values, the ECM model equation and the Nelder-Mead numerical optimization method
            to determine the RC parameters of the battery

            INPUT: ocv, terminal voltage, current and time data for a current pulse as well as initial estimates for RC parameters.
            OUTPUT: Optimal RC parameters to determine the internal resistance of the battery
            '''
            #use diff to determine the time interval
            time_diff = np.append(np.diff(time),np.diff(time)[-1])*3600

            def fit_ECM(r0,r1,c1):
                '''
                This function is nested in the optimization function. It determines the voltage output
                of the 1RC ECM model given RC parameters.

                INPUT: RC parameters
                OUTPUT: Associated voltage profile of ECM
                '''

                #create list to store the calculated voltage
                #and initiate variable to store v1
                fit_voltage = []
                ir1 = 0

                #loop through the length of the available data
                for i in range(len(ocv)):

                    #find the voltage across r1
                    ir1 = ir1 * math.exp(-time_diff[i] / (r1*c1)) + (current[i] * (1 - (math.exp(-time_diff[i] / (r1*c1)))))
                    vr1 = r1 * ir1

                    #combine to find the total voltage
                    v_new = ocv[i] - r0*current[i] - vr1

                    fit_voltage.append(v_new)

                return fit_voltage

            # nested function to calculate the rmse for given RC parameters
            def rmse(p):
                model = fit_ECM(p[0],p[1],p[2])
                mse = mean_squared_error(voltage, model)
                rmse = np.sqrt(mse)
                return rmse

            # determine the initial conditions and determine the bounds
            initial_variables = [init_r0,init_r1,init_c]
            bounds = [(0.0001, 1), (0.0001, 1), (5000, 30000)]

            # use the nelder mead method to find RC parameters that minimize the difference between expected and modelled voltage
            result = minimize(rmse, initial_variables, bounds=bounds, method='Nelder-Mead', options={'maxiter': 50000})
            optimized_variables = result.x

            return optimized_variables[0], optimized_variables[1], optimized_variables[2]

        #extract data for GITT pulses
        gitt_time = np.array(self.GITT["Time"])
        gitt_val = np.array(self.GITT["Terminal Voltage"])
        gitt_current = np.array(self.GITT["Current"])

        #record the start time of the gitt
        self.time = gitt_time[0]

        #determine the time intervals for the pulse windows
        windows = pulse_intervals(gitt_val)

        #extract a single pulse
        pulse_index1 = windows[0]
        pulse_index2 = windows[1]

        #use the found indexes to extract the data for the pulse
        pulse_time = np.array(gitt_time[pulse_index1:pulse_index2])
        pulse_voltage = np.array(gitt_val[pulse_index1:pulse_index2])
        pulse_current = np.array(gitt_current[pulse_index1:pulse_index2])
        pulse_resistance = estimate_resistance(pulse_voltage,pulse_current)

        #extract all data from discharge to find the ocv
        discharge_time = np.array(self.Discharge["Time"])
        discharge_val = np.array(self.Discharge["Terminal Voltage"])
        discharge_current = np.array(self.Discharge["Current"])

        #keep only the discharge section
        index = np.where(discharge_val == 2.5)[0]
        if index.size > 0:
            discharge_time = discharge_time[:index[0] + 1]
            discharge_val = discharge_val[:index[0] + 1]
            discharge_current = discharge_current[:index[0] + 1]

        #find the capacity throughout the slow discharge
        discharge_capacity = coulomb_counting(discharge_current,discharge_time)
        discharge_soc = discharge_capacity/np.max(discharge_capacity)

        #subtract the min value of the cumulative charge transferred from the max
        #during the full discharge to get the max capacity of cell at the time of the RPT
        max_capacity = np.max(discharge_capacity)-np.min(discharge_capacity)

        #as we are discharging at C/15 which is not ideal, we assume the terminal voltage is given
        #by (V = OCV - IR) we use initial estimates for the internal resistance to improve the pseudo ocv
        discharge_ocv = discharge_val + (pulse_resistance)*abs(np.mean(discharge_current))

        #create a data frame for the ocv curve and assign into a class attribute
        self.ocv_curve = pd.DataFrame({'OCV': discharge_ocv, 'SOC': discharge_soc})

        #estimate the ocv for the pulse under the assumption the ocv is the voltage after sufficient rest
        key_points = [0,
              np.where(np.abs(np.insert(np.diff(pulse_voltage), 0, 0)) > 0.005)[0][0],
              np.where(np.abs(np.insert(np.diff(pulse_voltage), 0, 0)) > 0.005)[0][1],
              len(pulse_voltage)-1]

        start = np.full(key_points[1], pulse_voltage[0])
        end = np.full(key_points[3]-key_points[2], pulse_voltage[-1])

        intermediate_x = np.arange(key_points[1], key_points[2]+1, step=1)
        middle = np.interp(intermediate_x, [key_points[1], key_points[2]], [pulse_voltage[key_points[0]], pulse_voltage[key_points[3]]])

        pulse_ocv = np.concatenate((start, middle, end))

        #use the parameter fitting function fit_param_values() to find the parameter values at the second pulse
        r0,r1,c1 = fit_param_values(pulse_ocv,pulse_voltage,pulse_current,pulse_time,pulse_resistance/2,pulse_resistance/3)

        #create dictionary to store the model parameters and return them
        parameters = {"R0":r0,
                    "R1":r1,
                    "C1":c1,
                    "Max Capacity":max_capacity}
        
        return parameters

    

    



