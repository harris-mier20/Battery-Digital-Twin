'''
This script calls in a lookup table, generated in 'lookup_table.py' and creates a 3D visualization of the table to view how
RC parameters vary with respect to operating conditions.
'''

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np

# call in the data for a desired RC parameter
val = pd.read_csv("Data/r0-lookup-table.csv")

# define the y and x axis as temperature and state of charge
#state of charge is the columns and temperature are rows of the data frame
y = pd.to_numeric(val["Temperature"])
x = val.columns[1:].astype(float)

# Extract RC parameter value at each point and convert to numeric array
data = val.iloc[:, 1:].values.astype(float)

# Create a 3D surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, data, cmap='viridis')

# Set labels for each axis
ax.set_xlabel('State of Charge (SOC)', fontsize=14)
ax.set_ylabel('Temperature /K', fontsize=14)
ax.set_zlabel('Ohmic Resistance (R0) /Ω', fontsize=14)
plt.show()





