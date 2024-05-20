'''
This script facilitates the running of multiple calendar aging simulations back to back by looping through a list
of operating conditions values. Each time the an object is created with the operating conditions and a path to a
folder to save the data from the simulation.
'''
from calendar_aging_simulation import Simulation

#define SOC values to run the simulations at
socs = [0.6, 0.7, 0.8]

#loop through the simulations and run at the given SOC
for soc in socs:

    #in this case the operating conditions do not change
    x = Simulation(soc,25,soc,25,"Data/" + f'{soc:.2f}'.replace('.', '')[:-1] + "-25")
    del x