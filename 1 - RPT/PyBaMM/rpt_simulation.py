'''
This script uses the PyBaMM library and creates an experiment to recreate the current profile of an RPT test. This test fully discharges the cell at
a constant discharge rate before recharging and running a series of pulses at defined SOC levels.

It loops through a set of predefined operating temperatures and saves the associated time, current, terminal voltage, OCV and discharge capacity in csv files.
'''

import pybamm as pb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Define properties of the simulation - a DFN model is used
pb.set_logging_level("NOTICE")
model = pb.lithium_ion.DFN()

# Load the Mohtat2020 parameter set for the simulation
param = pb.ParameterValues("Mohtat2020")

#Define the RPT experiment steps and the sampling period for which the data is saved at
experiment = pb.Experiment(
    [
        (
            "Charge at 0.5C until 4.2 V",
            "Hold at 4.2 V for 10 minutes",
            "Rest for 10 minutes",

            "Discharge at C/40 until 2.8V",
            "Hold at 2.8V for 10 minutes",
            "Rest for 10 minutes",
            
            "Charge at 0.5C until 4.2 V",
            "Hold at 4.2 V for 10 minutes",
            "Rest for 10 minutes",

            "Discharge at 2A for 80 seconds",
            "Rest for 20 minutes",

            "Discharge at 0.5C until 4.15 V",
            "Hold at 4.15 V for 20 minutes",
            "Rest for 10 minutes",

            "Discharge at 2A for 80 seconds",
            "Rest for 20 minutes",

            "Discharge at 0.5C until 4.1 V",
            "Hold at 4.1 V for 20 minutes",
            "Rest for 10 minutes",

            "Discharge at 2A for 80 seconds",
            "Rest for 20 minutes",

            "Discharge at 0.5C until 4.05 V",
            "Hold at 4.05 V for 20 minutes",
            "Rest for 10 minutes",

            "Discharge at 2A for 80 seconds",
            "Rest for 20 minutes",

            "Discharge at 0.5C until 4 V",
            "Hold at 4 V for 20 minutes",
            "Rest for 10 minutes",

            "Discharge at 2A for 80 seconds",
            "Rest for 20 minutes",

            "Discharge at 0.5C until 3.95 V",
            "Hold at 3.95 V for 20 minutes",
            "Rest for 10 minutes",

            "Discharge at 2A for 80 seconds",
            "Rest for 20 minutes",

            "Discharge at 0.5C until 3.9 V",
            "Hold at 3.9 V for 20 minutes",
            "Rest for 10 minutes",

            "Discharge at 2A for 80 seconds",
            "Rest for 20 minutes",

            "Discharge at 0.5C until 3.85 V",
            "Hold at 3.85 V for 20 minutes",
            "Rest for 10 minutes",

            "Discharge at 2A for 80 seconds",
            "Rest for 20 minutes",

            "Discharge at 0.5C until 3.8 V",
            "Hold at 3.8 V for 20 minutes",
            "Rest for 10 minutes",

            "Discharge at 2A for 80 seconds",
            "Rest for 20 minutes",

            "Discharge at 0.5C until 3.75 V",
            "Hold at 3.75 V for 20 minutes",
            "Rest for 10 minutes",

            "Discharge at 2A for 80 seconds",
            "Rest for 20 minutes",

            "Discharge at 0.5C until 3.7 V",
            "Hold at 3.7 V for 20 minutes",
            "Rest for 10 minutes",

            "Discharge at 2A for 80 seconds",
            "Rest for 20 minutes",

            "Discharge at 0.5C until 3.65 V",
            "Hold at 3.65 V for 20 minutes",
            "Rest for 10 minutes",

            "Discharge at 2A for 80 seconds",
            "Rest for 20 minutes",

            "Discharge at 0.5C until 3.6 V",
            "Hold at 3.6 V for 20 minutes",
            "Rest for 10 minutes",

            "Discharge at 2A for 80 seconds",
            "Rest for 20 minutes",

            "Discharge at 0.5C until 3.55 V",
            "Hold at 3.55 V for 20 minutes",
            "Rest for 10 minutes",

            "Discharge at 2A for 80 seconds",
            "Rest for 20 minutes",

            "Discharge at 0.5C until 3.5 V",
            "Hold at 3.5 V for 20 minutes",
            "Rest for 10 minutes",

            "Discharge at 2A for 80 seconds",
            "Rest for 20 minutes",

            "Discharge at 0.5C until 3.45 V",
            "Hold at 3.45 V for 20 minutes",
            "Rest for 10 minutes",

            "Discharge at 2A for 80 seconds",
            "Rest for 20 minutes"
        )
    ],
    period="5 seconds",
)

#define the temperatures, in Kelvin, to loop through and save
temp_c = np.array([10,20,30,40])
temp_k = temp_c + 273.15

#loop through all the kelvin temperatures
for temp in temp_k:

    #update the ambient temperature with the current temperature to simulate
    param.update({"Ambient temperature [K]": temp})

    # Run the simulation with the Casadi Solver
    sim = pb.Simulation(model, experiment=experiment, parameter_values=param)
    solution = sim.solve(solver=pb.CasadiSolver(mode="fast with events"))

    # Extract time, current, terminal voltage, OCV and discharge capacity from the solution
    time = solution["Time [h]"].entries
    current = solution["Current [A]"].entries
    terminal_voltage = solution["Terminal voltage [V]"].entries
    ocv = solution["Measured open circuit voltage [V]"].entries
    discharge_capacity = solution["Discharge capacity [A.h]"].entries

    # Create a DataFrame with pandas with recorded values
    df = pd.DataFrame({
        "Time": time,
        "Current": current,
        "Terminal Voltage": terminal_voltage,
        "Measured OCV": ocv,
        "Measured Discharge Capacity": abs(discharge_capacity)
    })

    #create a new directory to save the simulation result at this temperature
    relative_path = "data/" + str(temp)
    full_path = os.path.join(os.getcwd(), relative_path)
    
    if not os.path.exists(full_path):
        os.makedirs(full_path)

    #export the simulation result as a csv
    data_string = "data/" + str(temp) + "/" + str(temp) + "-rpt-test.csv"
    df.to_csv(data_string, index=False)

    # Plot the Terminal Voltage against time to check the simulation result
    plt.plot(time, terminal_voltage, color = "red")
    plt.xlabel("Time / h", fontsize = 14)
    plt.ylabel("Terminal Voltage / V", fontsize = 14)
    plt.title("RPT Simulation at Temperature = " + str(temp))
    plt.clf()