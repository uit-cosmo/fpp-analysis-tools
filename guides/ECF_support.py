# Some support functions for ECF_estimate_covariance.ipynb

import numpy as np
from scipy.optimize import Bounds, minimize
import fppanalysis as fa


def find_best_cf_var_step_acov_det(cf_var_len, P, CF, CF_jac, samples, **kwargs):
    # Attempt to find the best cf_var_step
    # as the one which minimizes the determinant of the asymptotic covariance.
    cf_var = lambda step: np.arange(1, cf_var_len + 1) * step
    minfun = lambda step: np.linalg.det(
        fa.asymptotic_covariance(cf_var(step), P, CF, CF_jac, samples)
    )

    return minimize(minfun, 1.0, bounds=((0.1, 10),), **kwargs)
    
def generator_gamma_norm(g, size=None):
    # Draw samples from the gamma distribution, normalized
    X = np.random.gamma(g, 1, size=size)

    return (X-g)/np.sqrt(g)
    
def generator_gamma_gauss_norm(g, e, size=None):
    # Draw samples from the gamma distribution with noise, normalized
    X = np.random.gamma(g, 1, size=size)
    Y = np.random.normal(scale = np.sqrt(e*g), size=size)
    Phi = X + Y
    
    return (Phi-g)/np.sqrt(g*(1+e))

class DistributionFamily:
    def __init__(self, name,
                 CF, CF_jac, generator, 
                 params, param_names, param_bounds):
        self.name = name
        
        self.CF = CF
        self.CF_jac = CF_jac
        self.generator = generator
        
        self.params = params
        self.param_names = param_names
        self.param_bounds = param_bounds
    
    def realization(self, samples):
        return self.generator(*self.params, size=samples)

    def change_params(self, new_params):
        self.params = new_params
        return self