'''
This script creates a function that is used in the 'run_simulation.py' script. This function takes in soc, current and
temperature and duration (in cycles) for 2 stages of degradation. It then uses this information to construct an aging experiment
with the real-world drive cycle and periodic RPT tests. It returns the PyBaMM experiment object and an array containing the cycle
numbers of the RPT test so only this data can be saved.
'''

import pybamm as pb
import pandas as pd
import numpy as np

#load in the current steps and durations for the bike journey data and save them as global variables
instructions = pd.read_csv("data/drive-cycle-steps.csv")
duration = np.array(instructions["Duration"])
current_steps = np.array(instructions["Current"])

def create_experiment(soc1,current1,temp1,cycles1,soc2,current2,temp2,cycles2,parameter_values):
    '''
    This function takes 2 values for average current, temperature and average state of charge where the fist value corresponds with conditions
    in the first phase of degradation that lasts a number of cycles defined by the cycle number and the second value corresponds with the second phase.
    The function returns the associated pybamm experiment object which can be used to run the simulation. It also returns the cycle numbers at which RPTs are performed
    so only that data can be saved.

    INPUT: average discharge current, SOC and temp for 2 stages of degradation and the number of cycles for each stage. Also takes in the pybamm parameter values
    to determine what voltage to charge to when targeting a specific state of charge.

    OUTPUT: Pybamm experiment object and array specifying what cycles to save data at (the RPT cycles)
    '''

    #extract the voltage cutoffs
    low_cutoff = parameter_values["Lower voltage cut-off [V]"]
    high_cutoff = parameter_values["Upper voltage cut-off [V]"]
    cell_capacity = parameter_values['Nominal cell capacity [A.h]']

    #define the number of journeys in a day and the number of days in a cycle
    journeys_per_charge = 3
    charge_per_cycle = 30

    #define the rest duration in hours
    rest = 7

    #determine the target soc and ocv during charging to achieve the desired average soc
    total_discharge1 = current1*np.sum(duration)*journeys_per_charge/3600
    target_soc1 = (soc1*cell_capacity+(total_discharge1/2))/cell_capacity
    if target_soc1 >= 1: target_soc1 = 1
    target_ocv1 = round(low_cutoff + target_soc1*(high_cutoff-low_cutoff),2)

    total_discharge2 = current2*np.sum(duration)*journeys_per_charge/3600
    target_soc2 = (soc2*cell_capacity+(total_discharge2/2))/cell_capacity
    if target_soc2 >= 1: target_soc2 = 1
    target_ocv2 = round(low_cutoff + target_soc2*(high_cutoff-low_cutoff),2)

    #scale the current steps so the average discharge current of a journey is specified
    #by the parameter values
    current_steps1 = current_steps*current1
    current_steps2 = current_steps*current2

    #create the templates for creating instructions with PyBaMM
    s = pb.step.string
    c = pb.step.current

    #create lists for all the instructions of the journeys, journey data is saved as a global variable
    journey1 = []
    journey2 = []

    #loop through the journey data and append current steps to the experiment with specified durations
    for i in range(len(duration)):
        journey1.append(c(current_steps1[i], duration = int(duration[i]), temperature = str(temp1) + "oC", period = 1))
        journey2.append(c(current_steps2[i], duration = int(duration[i]), temperature = str(temp2) + "oC", period = 1))

    #append the rest period to the drive cycle
    journey1.append(s("Rest for " + str(rest) + " hours", temperature = str(temp1) + "oC", period = 300))
    journey2.append(s("Rest for " + str(rest) + " hours", temperature = str(temp2) + "oC", period = 300))

    #repeat the journey for the number of journeys per charge
    journey1 = journey1 * journeys_per_charge
    journey2 = journey2 * journeys_per_charge

    #add the charge cycle to the start of the journey steps
    journey1.insert(0, s("Charge at C/2 until " + str(target_ocv1) + "V", temperature = str(temp1) + "oC", period = 10))
    journey2.insert(0, s("Charge at C/2 until " + str(target_ocv2) + "V", temperature = str(temp2) + "oC", period = 10))

    #repeat the profile for the number of charges per cycle
    journey1 = journey1 * charge_per_cycle
    journey2 = journey2 * charge_per_cycle

    #convert lists to tuples for a single cycle in pybamm
    journey1 = tuple(journey1)
    journey2 = tuple(journey2)
    
    #combined the journey with periodic RPT tests
    experiment = pb.Experiment((
            [
                (
                    s("Charge at C/2 until " + str(high_cutoff) + "V", temperature="25oC", period = 30),
                    s("Hold at " + str(high_cutoff) + "V until C/30", temperature="25oC", period = 30),
                    s("Discharge at C/10 until " + str(low_cutoff) + "V", temperature="25oC", period = 30),
                    s("Hold at " + str(low_cutoff) + "V until C/30", temperature="25oC", period = 30),
                    s("Charge at C/2 until 4V", temperature="25oC", period = 30),
                    s("Rest for 1 hour", temperature="25oC", period = 30),
                    s("Discharge at 0.5A for 30 minutes", temperature="25oC", period = 30),
                    s("Rest for 1 hour", temperature="25oC", period = 30),
                    s("Discharge at C/2 until " + str(low_cutoff) + "V", temperature="25oC", period = 30),
                )
            ] + [
                (
                    journey1
                )
            ]) * cycles1 + (
            [
                (
                    s("Charge at C/2 until " + str(high_cutoff) + "V", temperature="25oC", period = 30),
                    s("Hold at " + str(high_cutoff) + "V until C/30", temperature="25oC", period = 30),
                    s("Discharge at C/10 until " + str(low_cutoff) + "V", temperature="25oC", period = 30),
                    s("Hold at " + str(low_cutoff) + "V until C/30", temperature="25oC", period = 30),
                    s("Charge at C/2 until 4V", temperature="25oC", period = 30),
                    s("Rest for 1 hour", temperature="25oC", period = 30),
                    s("Discharge at 0.5A for 30 minutes", temperature="25oC", period = 30),
                    s("Rest for 1 hour", temperature="25oC", period = 30),
                    s("Discharge at C/2 until " + str(low_cutoff) + "V", temperature="25oC", period = 30),
                )
            ] + [
                (
                    journey2
                )
            ]) * cycles2, termination="95% capacity")
    
    #determine at what cycles data should be saved, these cycles correspond with RPT tests
    save_cycles = list(np.arange(1, 2*(cycles1+cycles2) + 1, 2))

    return experiment, save_cycles

