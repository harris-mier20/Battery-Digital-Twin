# Battery Digital Twin

This directory contains code associated with 4 work packages in the research and development of a battery digital twin framework. This includes:

1. Internal resistance measurements with reference performance tests.
2. Calendar aging simulations to extract physical relationships with non-parametric gaussian process regression (GPR).
3. Implementation of a real-world drive cycle into a PyBaMM simulation .
4. The generation of 3D lookup tables from simulation on the real-world profile at varying operating conditions in order to inform forecasts in a digital twin framework.

These work packages are reflected in the following folder structure:

```
├── Battery Digital Twin
│   ├── 1 - RPT
│   ├── 2 - Calendar Aging
│   ├── 3 - Real-world Aging
│   ├── 4 - Digital Twin Framework
```

<br><br>

## 1 - RPT

### Overview

<u>**Internal Resistance Measurements with Reference Performance Test**</u>

*Note: For this code to run, the working directory must be set to 'Battery Digital Twin/1 - RPT'*

This folder contains code associated with **simulating reference performance tests** on a battery cell to then perform health diagnostics to determine the internal resistance. This includes performing **least squares regression with a 1 RC ECM model.**

**Multiple operating conditions are simulated** to plot how ECM parameters vary with respect to SOC and ambient temperature. This **relationship is modelled with gaussian process regression** (GPR) to generate multi-dimensional lookup tables.

### File Structure

This folder is structured as follows:

```
├── 1 - RPT
│   ├── PyBaMM
|   │   ├── rpt_simulation.py
|   │   ├── post_processing.py
│   ├── Data
|   │   ├── 283.15
|   |   │   ├── Pulses
|   |   │   ├── 283.15-rpt-test.csv
|   |   │   ├── 283.15-ocv-curve.csv
|   │   ├── 293.15
|   |   │   ├── Pulses
|   |   │   ├── 293.15-rpt-test.csv
|   |   │   ├── 293.15-ocv-curve.csv
|   │   ├── 303.15
|   |   │   ├── Pulses
|   |   │   ├── 303.15-rpt-test.csv
|   |   │   ├── 303.15-ocv-curve.csv
|   │   ├── 313.15
|   |   │   ├── Pulses
|   |   │   ├── 313.15-rpt-test.csv
|   |   │   ├── 313.15-ocv-curve.csv
│   ├── ECM.py
│   ├── GPR.py
│   ├── lookup_table.py
│   ├── visualise_3D.py
│   ├── Figures
```

* **'rpt_simulation.py'** - This script simulates an RPT test with a full discharge and several current pulses. It loops through 4 different ambient temperature
and simulates and saves the simulation result as a csv in **'Data'** for each.

* **'post_processing.py'** - This script takes the data from the solution of the simulation, splits out individual pulses into separate csv files to be processes independently and performs coulomb counting to determine OCV with respect to SOC. It exports this ocv curve as a separate csv for each temperature.

* **'Data'** - This folder saves the simulation result of RPT tests at 4 different temperatures, with folder names dictating the ambient temperature in Kelvin. This data
is called in by **'lookup_table.py'** to fit RC parameters with respect to ambient temperature and generate lookup tables. Data saved includes the full RPT test output, individucal pulses at varying SOC and lookup tables for OCV, R0, R1 and C1 with respect to both temperature and SOC.

* **'ECM.py'** - This script provides helper functions to first make an estimate for 1RC ECM parameters given a current pulse and voltage profile. It then uses Scipy
tools to fit the ECM parameters with Nelder-Mead numerical optimization by minimizing the square difference between the expected terminal voltage and ECM output. These helper functions are used in **'lookup_table.py'**.

* **'GPR.py'** - This script provides helper functions to run non-parametric gaussian process regression to learn the relationship between RC parameters and operating conditions.These helper functions are used in **'lookup_table.py'**.

* **'lookup_table.py'** - This script calls in the pulse data at varying operating conditions to first construct independent 1 dimensional lookup tables for RC parameters with respect to SOC at discrete temperatures before combining these lookup tables with a secondary GPR process to create a single 2 dimensional lookup table for each RC parameter with respect to SOC and temperature. It calls in data from **'Data'** and helper functions from **'ECM.py'** and **'GPR.py'**.

* **'visualise_3D.py'** - This script calls in a lookup table, generated in **'lookup_table.py'** and creates a 3D visualization of the table to view how RC parameters vary with respect to operating conditions.

* **'Figures'** - This folder saves important figures that have been generated throughout the development process.

<br><br>

## 2 - Calendar Aging Simulations

### Overview

<u>**Calendar Aging Simulation and Analysis with Control Over SOC and Ambient Temperature**</u>

*Note: For this code to run, the working directory must be set to 'Battery Digital Twin/2 - Calendar Aging Simulations'*

This folder contains code associated with **simulating calendar aging** with parametric **control over the SOC of the battery and ambient temperature** during the aging process. the RPT test functionality developed in **'1 - RPT'** is used to determine state of health (SOH) over time and **extract the relationships between operating conditions and degradation** of the battery. These relationships are compared to known physical phenomenon, namely the anode half-cell voltage and the Arrhenius rate law.

Some empirical models are fit to the data with a monte carlo simulation as an **initial assessment of probabilistic forecast quality**. This is developed further in '4 - Digital Twin Framework'

### File Structure

This folder is structured as follows:

```
├── 2 - Calendar Aging Simulations
│   ├── PyBaMM
|   │   ├── calendar_aging_simulation.py
|   │   ├── extract_rpt.py
|   │   ├── run_simulation.py
│   ├── Data
|   │   ├── 04-55
|   │   ├── 08-25
|   │   ├── Varying SOC
|   │   ├── Varying Temp
|   │   ├── half-cell-voltage.csv
│   ├── GPR.py
│   ├── RPT.py
│   ├── extract_soh.py
│   ├── forecasting.py
│   ├── relationships.py
│   ├── compare_models.py
│   ├── Figures

```

* **'calendar_aging_simulation.py'** - This script creates a class for simulating a calendar aging profile. The class is created with arguments defining the SOC and temperature and number of cycles for the first stage of aging as well as arguments for a secondary stage facilitating the analysis of changing operating conditions. The class has one primary class method 'run_experiment()' which creates a pybamm experiment with the input arguments and saves data for all RPT tests in a single csv at a specific file location. This data is typically saved in **'Data'**.

* **'extract_rpt.py'** - This script creates a helper function to take a full data set of rpt tests and split it out into individual csv files, each containing a single pulse or a single full discharge. These are used to determine the SOH of the battery at a given time. This function is directly called within this script can be be adjusted depending on the file path of the data that needs to be extracted.

* **'run_simulation.py'** - This script facilitates the running of multiple calendar aging simulations back to back by looping through a list of operating conditions values. Each time the an object is created with the operating conditions and a path to a folder to save the data from the simulation.

* **'Data'** - Data from the output of simulations is saved here. Each simulation, with specific operating conditions, is saved to a dedicated folder in the format 'xx-yy' where xx corresponds to the state of charge of the simulation and yy corresponds to the temperature (e.g. 04-55 = 40% SOC and 55 deg C). Data for simulations with changing operating conditions would also be stored here. Each folder contains the full data set generated by the simulation, the individual full discharges and GITT pulses, as extracted by **'extract_rpt.py'** and a file named **'extracted.csv'** which contains the internal resistance and total available capacity vs time for the simulation. This is obtained with **'extract_soh.py'**. This folder also contains pre-processed health data for simulations with fixed temperature byt varying SOC (**'Varying SOC'**) and fixed SOC with varying temperature (**'Varying Temp'**). This data is used to extract relationships to compare to physical phenomena.

* **'half-cell-voltage.csv'** This csv contains data generated by PyBaMM for the Anode half-cell discharge voltage. This data is used to compare against and validate the relationship extracted using GPR for the degradation rate with respect to SOC. The rate of SEI growth is generally proportional to the anode half-cell voltage.

* **'GPR.py'** - This script provides helper functions to run non-parametric gaussian process regression to learn the relationship between degradation and operating conditions.These helper functions are used in **'physical_relationship.py'** to assess if the generated non-parametric relationships align with known physical phenomenon, namely the anode half-cell voltage and the Arrhenius rate law.

* **'RPT.py'** - This script creates a class that uses the coloumb counting and equivalent circuit modelling developed in **'1 - RPT'** to determine the SOH of the battery at a given time using data frames containing time,current and voltage data for a full discharge and a current pulse.

* **'extract_soh.py'** - This script uses the class created in **'RPT.py'** to process the rpt profiles that were performed throughout the aging simulation. For each RPT test, this script saves the time, total available capacity and internal resistance of the battery. It exports this data as a csv called **'extracted.csv'** and saves this data in the folder dedicated to the specific simulation.

* **'forecasting.py'** - This script implements an empirical model that was found to represent the nature of degradation well. It includes both a linear and non-linear factor. This model is fit to a subset of data points with a simulated measurement uncertainty to make a forecast for the remainder of the data points. This functionality is developed further in **'4 - Digital Twin Framework'**.

* **'relationships.py'** - This script uses pre-processed data, from **'Data'** sets for health against time for a series of calendar aging simulations at both fixed SOC and varying temperature and fixed temperature and varying SOC. It uses these data sets and fits an empirical degradation model to save the linear degradation rate with respect to operating conditions. It models this relationship with a GPR helper function from **'GPR.py'** and makes plots to compare against physical phenomenon such as the Arrhenius rate law and the anode potential.

* **'compare_models.py'** - This script calls in data for degradation under specific operating conditions, from **'Data'** and fits three models to the data, namely a linear model, a square root time model and an empirical model considering the the non-linear and linear phase of degradation, otherwise known as the 'degradation model'. Each fit is plotted and R scores are compared.

* **'Figures'** - This folder saves important figures that have been generated throughout the development process.

<br><br>

## 3 - Real World Simulation

### Overview

<u>**Processing and Implementation of a Real-world Electric Bike Drive Cycle in PyBaMM to Simulate Degradation Under Real World Conditions**</u>

*Note: For this code to run, the working directory must be set to 'Battery Digital Twin/3 - Real World Simulation'*

This folder contains code associated with the post processing of raw electric bike data to construct a drive cycle that is analogous to real world operating conditions. This drive cycle is implemented into a PyBaMM simulation with parametric control over the average SOC of the battery, the average discharge current during aging and the ambient temperature during aging. This folder also contains code for running and saving simulations at various operating conditions for later use in **' 4 - Digital Twin Framework'**.

### File Structure

This folder is structured as follows:

```
├── 3 - Real World Simulation
│   ├── Data
|   │   ├── ion-o-raw-data.csv
|   │   ├── drive-cycle-steps.csv
|   │   ├── 08-05-55
|   │   ├── 09-05-25
│   ├── PyBaMM
|   │   ├── create_experiment.py
|   │   ├── create_simulation.py
|   │   ├── run_simulation.py
|   │   ├── extract_rpt.py
│   ├── processing_raw_data.py
│   ├── RPT.py
│   ├── extract_soh.py
│   ├── Figures

```

* **'Data'** - This folder saves data that is used by PyBaMM to run simulations. It also contains folders dedicated to specific simulations, denoted by xx-yy-zz where xx represents the soc used in the simulation (e.g. 06 = 60% SOC), yy represents the average discharge current used in simulated journeys (e.g. 05 = 0.5A) and zz represents the average temperature used in the simulation (e.g. 55 = 55°C).

* **'ion-o-raw-data.csv'** - This csv file is a raw data file from the ION-O electric bike. Containing time, current voltage and temperature data for a series of bike journeys and rests. The current profile of a single bike journey is extracted and post processed in **'processing_raw_data.py'**.

* **'drive-cycle-steps.csv'** - This csv file is generated by **'processing_raw_data.py'** and contains data for current steps and the associated durations for a single journey from the ION-O bike data. This data can be read in by **'create_simulation.py'** where a PyBaMM simulation is constructed with this drive cycle, rest periods and RPT tests.

* **'create_experiment.py'** - This script creates a function that is used in the **create_simulation.py'** script. This function takes in soc, current and temperature and duration (in cycles) for 2 stages of degradation. It then uses this information to construct an aging experiment with the real-world drive cycle and periodic RPT tests. It returns the PyBaMM experiment object and an array containing the cycle numbers of the RPT test so only this data can be saved.

* **'create_simulation.py'** - This script create an class called **'Simulation'** that is used to create aging simulations with the real world drive cycle. The class is called in by the file **'run_simulation.py'** to run the simulation by creating objects with arguments for soc, current and temperature and duration (in cycles) for 2 stages of degradation. The class takes the input operating conditions and uses **'create_experiment.py'**  to construct the associated PyBaMM experiment. It saves data for the the RPT tests in a single csv at the specified folder location.

* **'run_simulation.py'** - This script facilitates the running of multiple real-world drive cycle aging simulations back to back by looping through a list of operating conditions values. Each time the an object is created with the operating conditions and a path to a folder to save the data from the simulation.

* **'extract_rpt.py'** - This script creates a helper function to take a full data set of rpt tests and split it out into individual csv files, each containing a single pulse or a single full discharge. These are used to determine the SOH of the battery at a given time. This function is directly called within this script can be be adjusted depending on the file path of the data that needs to be extracted.

* **'processing_raw_data.py'** - This script takes in the current and time data from **'ion-o-raw-data.csv'** and smooths and discretizes it before reformatting it as a series of current steps and associated durations. It exports the reformatted data as a csv saved as **'drive_cycle_steps.csv'**.

* **'RPT.py'** - This script creates a class that uses the coloumb counting and equivalent circuit modelling developed in **'1 - RPT'** to determine the SOH of the battery at a given time using data frames containing time,current and voltage data for a full discharge and a current pulse.

* **'extract_soh.py'** - This script uses the class created in **'RPT.py'** to process the rpt profiles that were performed throughout the aging simulation. For each RPT test, this script saves the time, total available capacity and internal resistance of the battery. It exports this data as a csv called **'extracted.csv'** and saves this data in the folder dedicated to the specific simulation.

* **'Figures'** - This folder saves important figures that have been generated throughout the development process.

<br><br>

## 4 - Digital Twin Framework

### Overview

<u>**Combination of all frameworks to generate multi-dimensional lookup tables and probabilistically predict degradation under different operating conditions**</u>

*Note: For this code to run, the working directory must be set to 'Battery Digital Twin/4 - Digital Twin Framework'*

This folder contains code associated with putting together the framework of the digital twin. It takes data from real-world drive cycle aging simulations with multiple sampled operating conditions to then use GPR to construct 2 dimensional lookup tables for degradation model coefficients with respect to operating conditions. These lookup tables are then used by the digital twin to make forecasts. The **'digital_twin.py'** file creates a class to store the features of the digital twin, including making probabilistic health forecasts and making deterministic health forecasts for changes in operating conditions (what-if questions) using only the lookup tables and previously seen health data for the battery. The framework is validated with simulated test data which is saved in the **'Data'** folder.

### File Structure

This folder is structured as follows:

```
├── 4 - Digital Twin Framework
│   ├── Data
|   │   ├── 0.5A Simulations
|   │   ├── 0.5A Test Data
|   │   ├── A-lookup.csv
|   │   ├── B-lookup.csv
|   │   ├── C-lookup.csv
|   │   ├── D-lookup.csv
│   ├── GPR.py
│   ├── generate_lookup_tables.py
│   ├── visualise_lookup_tables.py
│   ├── digital_twin.py
│   ├── test_digital_twin.py
│   ├── Figures
```

* **'Data'** - This folder contains data for simulation outputs, specifically extracted SOH data with respect to time for various operating conditions. This data is generated using the simulations in **'3 - Real World Simulation'**. This folder also contains the 2 dimensional lookup tables that are generated in **'generate_lookup_tables.py'**. These lookup tables are later called in by the **Digital Twin** class created in **'digital_twin.py'**.

* **'0.5A Simulations'** - This folder contains extracted simulation results with the Capacity and Internal Resistance measured for a battery, under real world conditions, over time. Files in this format are named with the convention 'xx-yy-zz.csv' where 'xx' refers to the soc used in the simulation (e.g. 06 = 60% SOC), 'yy' refers to the average discharge current during a bike journey (e.g. 05 = 0.5A) and 'zz' refers to the ambient temperature (e.g. 325 = 32.5 °C)

* **'0.5A Test Data'** - This folder contains data used to test the digital twin framework. For 3 sets of operating conditions it contains 3 extracted health data files for different scenarios. For example for aging at 78% SOC, 0.5A and 30 °C it contains a file where these operating conditions did not change, a file where both temperature and SOC increased at a specified time and a file where they both decrease at a specified time. A table summarizing the characteristics of these simulations is found below.

* **'GPR.py'** - This script provides helper functions to run non-parametric gaussian process regression to learn the relationship between degradation and operating conditions under the real world operating conditions.These helper functions are used in **'generate_lookup_tables.py'** to construct multi-dimensional lookup tables for degradation function coefficients with respect to operating conditions.

* **'generate_lookup_tables.py'** - This script uses the simulation outputs at sampled operating conditions to determine the degradation model coefficients at these operating conditions before fitting GPR curves in 2 dimensions (variations of each coefficient with respect to both SOC and Temperature). For each coefficient, it creates a 2D lookup table which is exported as a csv and saved in **'Data'**.

* **'visualise_lookup_tables.py'** - This script takes a csv file of a lookup table, that was generated in **'generate_lookup_tables.py'**, and creates a 3D graph to visualise the relationships between operating conditions and degradation model coefficients.

* **'digital_twin.py'** - This script creates a class to store the functionality of a digital twin. This includes taking test data generated from **'3 - Real World Simulation'** for aging under real world conditions. It makes probabilistic forecasts at a specified time for the health at the end of the simulation, assuming operating conditions do not change and assesses this forecast accuracy with the continuous ranked probability score. It also makes deterministic predictions for the trajectory of health for a specified change in operating conditions and assess this accuracy with the mean absolute percentage error score. This class is called in by the **'test_digital_twin.py'** script to test it with a series of test data sets.

* **'test_digital_twin.py'** - This script calls in the class created in **'digital_twin.py'** and uses it to create probabilistic forecasts and ask what if questions for simulated scenarios that are outlined in the table below and stored in **'0.5A Test Data'**. The **'DigitalTwin'** class is used to assess the probabilistic forecast error of the digital twin while assessing the error in predicting the future degradation trajectory after a change in operating conditions using the lookup tables generated with GPR in **'generate_lookup_tables.py'**.

* **'Figures'** - This folder saves important figures that have been generated throughout the development process.

### Test Data Scenarios

Test data was generated with simulations of different aging scenarios and changes in operating conditions to assess the performance of the digital twin. These files are found in "Data>0.5A Test Data". The scenarios are found below.

```
|            | Set 1                                                 | Set 2                                                 | Set 3                                               |
|------------|-------------------------------------------------------|-------------------------------------------------------|-----------------------------------------------------|
| Starting   | 78% SOC, 30 °C                                        | 75% SOC, 35 °C                                        | 81% SOC, 35 °C                                      |
| Scenario 1 | No Changes - 078-05-30-base.csv                       | No Changes - 075-05-35-temp-base.csv                  | No Changes - 081-05-35-soc-base.csv                 |
| Scenario 2 | Change to 89.7% SOC, 34.5 °C - 078-05-30-up.csv       | Change to 75% SOC, 43.75 °C - 075-05-35-temp-up.csv   | Change to 89.1% SOC, 35 °C - 081-05-35-soc-up.csv   |
| Scenario 3 | Change to 66.3% SOC, 25.5 °C - 078-05-30-down.csv     | Change to 75% SOC, 26.25 °C - 075-05-35-temp-down.csv | Change to 72.9% SOC, 35 °C - 081-05-35-soc-down.csv | 
```