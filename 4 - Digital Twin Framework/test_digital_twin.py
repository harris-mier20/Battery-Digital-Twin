'''
This script calls in the class created in **'digital_twin.py'** and uses it to create probabilistic forecasts and ask what if questions
for simulated scenarios that are outlined in the table below and stored in '0.5A Test Data'. The 'DigitalTwin' class is used to
assess the probabilistic forecast error of the digital twin while assessing the error in predicting the future degradation trajectory after
a change in operating conditions using the lookup tables generated with GPR in 'generate_lookup_tables.py'.
'''

from digital_twin import DigitalTwin
import pandas as pd

#import all the test data
baseline_data = pd.read_csv("Data/0.5A Test Data/078-05-30-base.csv")
increasing_data = pd.read_csv("Data/0.5A Test Data/078-05-30-base.csv")
decreasing_data = pd.read_csv("Data/0.5A Test Data/078-05-30-base.csv")
data = [baseline_data,increasing_data,decreasing_data]

#define the operating conditions and save in a dictionary, these can be found from the README.md file
operating_conditions = {"baseline": [0.78,0.5,30],
                        "increasing": [0.897,0.5,34.5],
                        "decreasing": [0.663,0.5,25.5]}

#load in the lookup tables
A_table = pd.read_csv("Data/A-lookup.csv")
B_table = pd.read_csv("Data/B-lookup.csv")
C_table = pd.read_csv("Data/C-lookup.csv")
D_table = pd.read_csv("Data/D-lookup.csv")
tables = [A_table, B_table, C_table, D_table]

#create the object for testing the digital twin framework
test = DigitalTwin(data, operating_conditions, tables)

#make plots for the probabilistic forecasts
test.get_baseline_cdf_vals()
test.cdf_baseline_plot()
test.plot_probabilistic_baseline()

#make what-if forecasts and calculate the NRMSE (normalized RMSE considering the range of likely values of SOH)
test.what_if()
test.what_if_error()
