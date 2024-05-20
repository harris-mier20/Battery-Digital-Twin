'''
Uses the aging simulation object to simulate multiple operating conditions back to back
'''
from create_simulation import Simulation

folder_location = "Data/08-05-55"
x = Simulation(0.8,0.5,55,0.8,0.5,55,folder_location)
del x

folder_location = "Data/09-05-25"
x = Simulation(0.9,0.5,25,0.9,0.5,25,folder_location)
del x





