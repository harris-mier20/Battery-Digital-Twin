'''
This script provides helper functions to run non-parametric gaussian process regression to learn the relationship
between degradation and operating conditions under the real world operating conditions.These helper functions are used
in 'generate_lookup_tables.py' to construct multi-dimensional lookup tables for degradation function coefficients with
respect to operating conditions.
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

class SquaredExponentialKernel:
    def __init__(self, sigma_f: float = 1, length: float = 1):
        self.sigma_f = sigma_f
        self.length = length

    def __call__(self, argument_1: np.array, argument_2: np.array) -> float:
        return float(self.sigma_f *
                     np.exp(-(np.linalg.norm(argument_1 - argument_2)**2) /
                            (2 * self.length**2)))
    
# Helper function to calculate the respective covariance matrices
def cov_matrix(x1, x2, cov_function) -> np.array:
    return np.array([[cov_function(a, b) for a in x1] for b in x2])

class GPR:
    def __init__(self,
                 data_x: np.array,
                 data_y: np.array,
                 covariance_function=SquaredExponentialKernel(length = 1),
                 white_noise_sigma: float = 0):
        self.noise = white_noise_sigma
        self.data_x = data_x
        self.data_y = data_y
        self.covariance_function = covariance_function

        # Store the inverse of covariance matrix of input (+ machine epsilon on diagonal) since it is needed for every prediction
        self._inverse_of_covariance_matrix_of_input = np.linalg.inv(
            cov_matrix(data_x, data_x, covariance_function) +
            (3e-7 + self.noise) * np.identity(len(self.data_x)))

        self._memory = None

    # function to predict output at new input values. Store the mean and covariance matrix in memory.

    def predict(self, at_values: np.array) -> np.array:
        k_lower_left = cov_matrix(self.data_x, at_values,
                                  self.covariance_function)
        k_lower_right = cov_matrix(at_values, at_values,
                                   self.covariance_function)

        # Mean.
        mean_at_values = np.dot(
            k_lower_left,
            np.dot(self.data_y,
                   self._inverse_of_covariance_matrix_of_input.T).T).flatten()

        # Covariance.
        cov_at_values = k_lower_right - \
            np.dot(k_lower_left, np.dot(
                self._inverse_of_covariance_matrix_of_input, k_lower_left.T))

        # Adding value larger than machine epsilon to ensure positive semi definite
        cov_at_values = cov_at_values + 3e-7 * np.ones(
            np.shape(cov_at_values)[0])

        var_at_values = np.diag(cov_at_values)

        self._memory = {
            'mean': mean_at_values,
            'covariance_matrix': cov_at_values,
            'variance': var_at_values
        }
        return mean_at_values
    

def fit_GPR(model, x):

    # use the GPR model to predict the value of y given a value of x
    mean = model.predict(x)

    # determine the standard deviation as a measure of uncertainty
    std = np.sqrt(model._memory['variance'])

    return mean,std

def run_GPR(x_values,y_values,x,length=1,sigma=1, noise = 0):

    # fit the GPR model with the provided parameters for the squared exponential kernel
    model = GPR(x_values, y_values, covariance_function=SquaredExponentialKernel(length = length, sigma_f=sigma), white_noise_sigma=noise)

    return fit_GPR(x=x, model=model)