'''
This script creates a class for simulating a calendar aging profile. The class is created with arguments defining the SOC and temperature and
number of cycles for the first stage of aging as well as arguments for a secondary stage facilitating the analysis of changing operating conditions.
The class has one primary class method 'run_experiment()' which creates a pybamm experiment with the input arguments and saves data for all RPT tests
in a single csv at a specific file location. This data is typically saved in 'Data'.
'''

import pybamm as pb
import numpy as np
import pandas as pd

class Simulation:
    '''
    Class takes a SOC and Temperature and runs an SEI simulation on the Chen2020 cell
    This is calendar aging at the fixed SOC with a series of RPT tests with slow discharge and pulses
    to determine the maximum available capacity and ECM resistance at the time of the rpt test
    '''
    def __init__(self,soc1,temp1,soc2,temp2,path):

        #define the operating conditions for the first stage
        self.soc1 = soc1
        self.temp1 = temp1

        #define the operating conditions after operating conditions change
        self.soc2 = soc2
        self.temp2 = temp2

        self.path = path

        #define the number of rpt tests to record for both stages
        #this means operating conditions can change after a certain point
        self.cycles1 = 20
        self.cycles2 = 20

        #run the simulation and save the results as a single csv file
        self.data = self.run_experiment()

    def run_experiment(self):
        '''
        Uses defined class variables to create and run a calendar aging simulations. Exports a data frame that
        contains time, current, voltage and ambient temperature for all rpt tests throughout the simulation
        '''
    
        def target_voltage(soc,low,high):
            '''
            returns a target voltage given a target soc and the high and low bounds for the cell

            INPUT: target soc, lower and upper voltage limits for the simulated cell.
            OUTPUT: target cell voltage to achieve desired SOC
            '''
            voltage = low+soc*(high-low)
            return voltage

        # load the OKane2022 battery parameter set
        parameter_values = pb.ParameterValues("OKane2022")

        #extract the voltage cutoffs to assist with determining the target voltage for a given SOC
        low_cutoff = parameter_values["Lower voltage cut-off [V]"]
        high_cutoff = parameter_values["Upper voltage cut-off [V]"]

        #define the solver settings
        solver = pb.CasadiSolver(mode="fast", return_solution_if_failed_early=True)

        #calculate target voltages to achieve the desired SOC
        hold_voltage_1 = str(round(target_voltage(self.soc1,low_cutoff,high_cutoff),3))
        hold_voltage_2 = str(round(target_voltage(self.soc2,low_cutoff,high_cutoff),3))

        s = pb.step.string

        #create the experiment with calendar aging and periodic RPT tests
        experiment = pb.Experiment((
            [
                (
                    s("Charge at C/2 until " + str(high_cutoff) + "V", temperature="25oC"),
                    s("Hold at " + str(high_cutoff) + "V until C/30", temperature="25oC"),
                    s("Discharge at C/10 until " + str(low_cutoff) + "V", temperature="25oC"),
                    s("Hold at " + str(low_cutoff) + "V until C/30", temperature="25oC"),
                    s("Charge at C/2 until 4V", temperature="25oC"),
                    s("Rest for 1 hour", temperature="25oC"),
                    s("Discharge at 0.5A for 30 minutes", temperature="25oC"),
                    s("Rest for 1 hour", temperature="25oC"),
                    s("Discharge at C/2 until " + str(low_cutoff) + "V", temperature="25oC"),
                )
            ] + [
                (
                    s("Charge at C/2 until "+ hold_voltage_1 +"V", temperature=str(self.temp1) + "oC"),
                    s("Rest for 500 hours", temperature=str(self.temp1) + "oC")
                )
            ]) * self.cycles1 + (
            [
                (
                    s("Charge at C/2 until " + str(high_cutoff) + "V", temperature="25oC"),
                    s("Hold at " + str(high_cutoff) + "V until C/30", temperature="25oC"),
                    s("Discharge at C/10 until " + str(low_cutoff) + "V", temperature="25oC"),
                    s("Hold at " + str(low_cutoff) + "V until C/30", temperature="25oC"),
                    s("Charge at C/2 until 4V", temperature="25oC"),
                    s("Rest for 1 hour", temperature="25oC"),
                    s("Discharge at 0.5A for 30 minutes", temperature="25oC"),
                    s("Rest for 1 hour", temperature="25oC"),
                    s("Discharge at C/2 until " + str(low_cutoff) + "V", temperature="25oC"),
                )
            ] + [
                (
                    s("Charge at C/2 until "+ hold_voltage_2 +"V", temperature=str(self.temp2) + "oC"),
                    s("Rest for 500 hours", temperature=str(self.temp2) + "oC")
                )
            ]) * self.cycles2, period="30 seconds", termination="95% capacity")


        #set the logging level of pybamm
        pb.set_logging_level("INFO")

        #define the lithium ion dfn model with only SEI and lithium plating
        model = pb.lithium_ion.DFN(
            {
                "SEI": "solvent-diffusion limited",
                "SEI porosity change": "true",
                "lithium plating": "partially reversible"
            }
        )

        #run the simulation and only save data for the RPT tests
        sim = pb.Simulation(model, experiment=experiment, parameter_values=parameter_values)
        solution = sim.solve(solver=solver, save_at_cycles=list(np.arange(1, 2*(self.cycles1+self.cycles2) + 1, 2)))

        # Extract time, current, and terminal voltage from the solution
        time = solution["Time [h]"].entries
        current = solution["Current [A]"].entries
        terminal_voltage = solution["Terminal voltage [V]"].entries
        temperature = solution['Ambient temperature [K]'].entries

        # Create a DataFrame with recorded values
        df = pd.DataFrame({
            "Time": time,
            "Current": current,
            "Terminal Voltage": terminal_voltage,
            "Ambient Temperature": temperature
        })

        path = self.path + "/full.csv"
        df.to_csv(path, index=False)