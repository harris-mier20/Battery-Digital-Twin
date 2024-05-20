'''
Loops through all the available pulses at different SOC to generate more complex lookup tables.
Fits GPR to each discrete temperature point. Then GPR between these temperature points.
A 3D graph for each variable can be drawn.
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ECM import fit_ECM
from GPR import GPR, SquaredExponentialKernel
import os

def get_RC_vals(temp):
    '''
    This function uses saved pulse RPT data at a given temperature and returns the associated ECM parameters
    with respect to SOC

    INPUT: temperature of interest as string.
    OUTPUT: dictionary with RC parameters with respect to SOC
    '''

    #define the directory to access the data
    directory = "Data/" + str(temp)
    pulse_directory = directory + "/Pulses"

    #get the number of pulses for each temperature
    full_path = os.path.join(os.getcwd(), pulse_directory)
    n_pulses = len(os.listdir(full_path))

    #generate empty arrays to save the R0 values with respect to SOC
    soc_vals = []
    r0_vals = []
    r1_vals = []
    c1_vals = []

    #use the pulses to fit RC parameters
    for i in range(n_pulses-1):

        #load the pulse
        pulse_name = directory + "/Pulses/" + str(i+1) + ".csv"
        pulse = pd.read_csv(pulse_name)
        soc = pulse["Estimate SOC"][0]

        #call in the helper function to fit ECM parameters
        rc_params = fit_ECM(pulse)
        rmse = rc_params["RMSE"]
        
        #check if the rmse is good enough
        if (rmse < 0.001):

            #append all the parameters at each soc
            soc_vals.append(soc)
            r0_vals.append(rc_params["R0"])
            r1_vals.append(rc_params["R1"])
            c1_vals.append(rc_params["C1"])

    #combine the results into a dictionary and return the dictionary
    results = {"SOC":soc_vals,"R0":r0_vals,"R1":r1_vals,"C1":c1_vals}
    return results

def temp_table(temp):
    '''
    # This function takes in a temperature (that has been used to simulate an RPT test), determines the RC values with respect
    # to SOC by separately analyzing each extracted pulse before fitting a GPR model to estimate the RC parameter value wrt SOC.

    # INPUT: temperature value (must be a valid temperature corresponding to a folder in 'Data')
    # OUTPUT: A data frame containing the RC parameters with respect to SOC for the input temperature.
    '''

    #call the get_RC_vals function to fit the RC parameters to the data for that temperature
    RC_vals = get_RC_vals(temp)

    #extract SOC, R0, R1 and C1   
    soc_vals = RC_vals["SOC"]
    r0_vals = RC_vals["R0"]
    r1_vals = RC_vals["R1"]
    c1_vals = RC_vals["C1"]

    #fit GPR models for each parameter
    r0_model = GPR(np.array(soc_vals),np.array(r0_vals),
                   covariance_function=SquaredExponentialKernel(length = 0.6, sigma_f=0.3),white_noise_sigma=0)
    r1_model = GPR(np.array(soc_vals),np.array(r1_vals),
                   covariance_function=SquaredExponentialKernel(length = 0.9, sigma_f=0.3),white_noise_sigma=0)
    c1_model = GPR(np.array(soc_vals),np.array(c1_vals),
                   covariance_function=SquaredExponentialKernel(length = 0.2, sigma_f=0.1), white_noise_sigma=0)
    
    #import the OCV SOC data for the temperature to start the lookup table
    filepath = "data/" + str(temp) + "/" + str(temp) + "-ocv-curve.csv"
    table = pd.read_csv(filepath)

    #append the data to the lookup table using the GPR models
    table["R0"] = r0_model.predict(table["SOC"])
    table["R1"] = r1_model.predict(table["SOC"])
    table["C1"] = c1_model.predict(table["SOC"])

    return table

def temp_relationship(soc_val,selected_parameter,tables,continuous_temp):
    '''
    This function takes a soc value and the name of an RC parameter of interest (e.g. R0) and fits a GPR relationship
    to that RC parameter with respect to temperature.

    INPUT: An SOC value of interest, the name of the RC parameter of interest, all the data frames containing RC parameters wrt
    SOC and the range of temperature values to fit the GPR to.

    OUTPUT: an array of predicted RC parameter values associated with the input temperature values.
    '''

    #create arrays to store the ground truths
    temps = [283.15, 293.15, 303.15, 313.15]
    vals = []

    #loop through those temperatures and add the R0 value that corresponds to the defined soc
    for table in tables:

        #find the selected RC parameter at the selected SOC for each temperature
        val = table.loc[table.index[table['SOC'] == soc_val].tolist()[0], selected_parameter]
        vals.append(val)
    
    #fit the a GPR model to continuous values of temperature
    model = GPR(np.array(temps),np.array(vals),
                   covariance_function=SquaredExponentialKernel(length = 50, sigma_f=0.3),white_noise_sigma=0)

    #return an array with the continuous RC parameter value wrt temperature at the defined SOC
    return model.predict(continuous_temp)   

#create lookup tables for each temperature using the helper function
table_283 = temp_table(283.15)
table_293 = temp_table(293.15)
table_303 = temp_table(303.15)
table_313 = temp_table(313.15)
tables = [table_283, table_293, table_303, table_313]

#determine all the soc values and temperature values for the psuedo continuous lookup table
continuous_soc = list(np.intersect1d(np.intersect1d(np.intersect1d(table_283["SOC"], table_293["SOC"]), table_303["SOC"]), table_313["SOC"]))
continuous_temp = np.arange(280,315,1)

#create a template dataframe to store the lookup tables
columns = continuous_soc[:]
columns.insert(0,"Temperature")
df = pd.DataFrame(columns=columns)

#create a data frame for R0
r0 = df
r0["Temperature"] = continuous_temp

#loop through all the soc values and find the R0 parameters with respect to temperature
for soc in continuous_soc:
    r0[soc] = temp_relationship(soc,"R0",tables,continuous_temp)

#export the table
r0.to_csv("data/r0-lookup-table.csv", index=False)

#create a data frame for R1
r1 = df
r1["Temperature"] = continuous_temp

#loop through all the soc values and find the R0 parameters with respect to temperature
for soc in continuous_soc:
    r1[soc] = temp_relationship(soc,"R1",tables,continuous_temp)

#export the table
r1.to_csv("data/r1-lookup-table.csv", index=False)

#create a data frame for C1
c1 = df
c1["Temperature"] = continuous_temp

#loop through all the soc values and find the R0 parameters with respect to temperature
for soc in continuous_soc:
    c1[soc] = temp_relationship(soc,"C1",tables,continuous_temp)

#export the table
c1.to_csv("data/c1-lookup-table.csv", index=False)

#create a data frame for OCV
ocv = df
ocv["Temperature"] = continuous_temp

#loop through all the soc values and find the R0 parameters with respect to temperature
for soc in continuous_soc:
    ocv[soc] = temp_relationship(soc,"OCV",tables,continuous_temp)

#export the table
ocv.to_csv("data/ocv-lookup-table.csv", index=False)