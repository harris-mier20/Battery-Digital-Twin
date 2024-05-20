'''
This script create an class called 'Simulation' that is used to create aging simulations with the real world drive cycle.
The class is called in by the file 'run_simulation.py' to run the simulation by creating objects with arguments for soc,
current and temperature and duration (in cycles) for 2 stages of degradation. The class takes the input operating conditions
and uses 'create_experiment.py'  to construct the associated PyBaMM experiment. It saves data for the the RPT tests in a
single csv at the specified folder location.
'''

import pybamm as pb
import pandas as pd
from create_experiment import create_experiment

class Simulation:
    '''
    This class takes SOC, discharge current and temperature values for 2 stages of degradation and runs an SEI simulation on the OKane2022 cell.
    The experiment is created with the create_experiment function that utilizes the real world drive cycle data. When initializing this class,
    the run_experiment() function is called. This runs the PyBaMM simulation and saves current, voltage and time data for all the RPT tests in a
    single file in the specified folder location.
    '''
    def __init__(self,soc1,current1,temp1,soc2,current2,temp2,path):

        #save variables for operating conditions during degradation in the fist phase
        self.soc1 = soc1
        self.current1 = current1
        self.temp1 = temp1

        # save variables for the second phase
        self.soc2 = soc2
        self.current2 = current2
        self.temp2 = temp2

        #save the file path to the folder dedicated to the simulation result
        self.path = path

        #define the number of rpt tests to record for both phases of degradation
        self.cycles1 = 12
        self.cycles2 = 44

        #run the simulation to get all the data as csv file
        self.run_experiment()

    def run_experiment(self):
        '''
        This function creates and runs the pybamm experiment. When called it runs the simulation it runs the PyBaMM simulation and saves current,
        voltage and time data for all the RPT tests in a single file in the specified folder location. 

        INPUT: class attributes for operating conditions
        OUTPUT: None
        '''

        #define the parameter values
        parameter_values = pb.ParameterValues("OKane2022")

        #create the experiment steps with the create_experiment.py file
        experiment,save_cycles = create_experiment(self.soc1,self.current1,self.temp1,self.cycles1,
                                       self.soc2,self.current2,self.temp2,self.cycles2,parameter_values)

        #define the solver settings
        solver = pb.CasadiSolver(mode="safe",dt_max = 0.01, return_solution_if_failed_early=True)

        #set the logging level of pybamm
        pb.set_logging_level("INFO")

        #define the lithium ion dfn model with SEI degradation
        model = pb.lithium_ion.DFN(
            {
                "SEI": "solvent-diffusion limited",
                "SEI porosity change": "true",
                "lithium plating": "partially reversible"
            }
        )

        #run the simulation
        sim = pb.Simulation(model, experiment=experiment, parameter_values=parameter_values)
        solution = sim.solve(solver=solver,save_at_cycles=save_cycles)

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

        #save the data at the specified folder location
        path = self.path + "/full.csv"
        df.to_csv(path, index=False)