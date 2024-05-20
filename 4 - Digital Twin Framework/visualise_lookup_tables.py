'''
This script takes a csv file of a lookup table, that was generated in 'generate_lookup_tables.py', and
creates a 3D graph to visualise the relationships between operating conditions and degradation model coefficients.
'''

import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import numpy as np

# call in the lookup table for the B parameter
val = pd.read_csv("Data/B-lookup.csv")

# define the y and x axis as temperature and state of charge
# state of charge is the columns and temperature are rows of the data frame
y = pd.to_numeric(val["Temperature"])
x = val.columns[2:].astype(float)

# Extract the B parameter at every point for 0.5A
data = val.iloc[:, 2:].values.astype(float)

# Create a 3D surface plot for 0.5A
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(x, y)
surf1 = ax.plot_surface(X, Y, data, alpha=1, cmap='viridis')

# Set labels for each axis
ax.set_xlabel('State of Charge (SOC)', fontsize = 14)
ax.set_ylabel('Temperature / °C', fontsize = 14)
ax.set_zlabel('Linear Degradation Rate Coefficient', fontsize = 14)

plt.legend([surf1], ['Average Discharge Current= 0.5 A'], loc='best')
plt.show()





