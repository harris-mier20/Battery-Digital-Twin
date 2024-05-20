'''
This script uses the class created in 'RPT.py' to process the rpt profiles that were completed throughout the aging simulation.
For each RPT test, this script saves the time, total available capacity and internal resistance of the battery. It exports this data as a
csv called 'extracted.csv' and saves this data in the folder dedicated to the specific simulation.
'''

import pandas as pd
import matplotlib.pyplot as plt
from RPT import RPT

#specify the folder location for the simulation
file_name = "Data/04-55"

#create arrays to store data to plot
time = []
capacity = []
resistance = []

#loop through the number of available data points
for i in range(20):

    #define the data file names
    data = [file_name + "/discharge-" + str(i) + ".csv",
            file_name + "/gitt-" + str(i) + ".csv"]
            
    #load the data
    Discharge = pd.read_csv(data[0])
    GITT = pd.read_csv(data[1])

    #create the RPT object with the full discharge and pulse data
    rpt = RPT(Discharge,GITT)

    #save the time, capacity and internal resistance at the time of the RPT test
    time.append(rpt.time)
    capacity.append(rpt.capacity)
    resistance.append(rpt.ecm_resistance)

#Plot the capacity data to check
plt.plot(time,capacity, color = "blue")
plt.xlabel('Time /h', fontsize = 14)
plt.ylabel('Capacity /Ah', fontsize = 14)
plt.title('Total Available Capacity Over Time')
plt.show()

#save the health data as single data frame
df = pd.DataFrame({'Time': time,
                'Capacity': capacity,
                'Resistance': resistance
})

#export the health data as a csv in the file location for the simulation
df.to_csv(file_name + "/extracted.csv", index=False)