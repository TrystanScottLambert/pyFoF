"""Survey class to keep all survey properties together."""

import pandas as pd
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import quad


class Survey:
    """Class representing an individual survey."""
    def __init__(self, data: pd.DataFrame, cosmology: FlatLambdaCDM,
                 apparent_mag_limit: float,
                 fiducial_velocity:float = 1000.,
                 alpha = -1.02,
                 m_star = -24.2,
                 phi_star = 0.0108):

        """Initializing"""
        self.apparent_magnitude_limit = apparent_mag_limit
        self.fiducial_velocity = fiducial_velocity
        self.data_frame = data
        self.cosmology = cosmology
        self.alpha = alpha
        self.m_star = m_star
        self.phi_star = phi_star
        self.integral = quad(self.shecter_function, -np.inf, self.absolute_mag_lim)

    @property
    def absolute_mag_lim(self):
        """Absolute magnitude limit."""
        abs_mag_lim = self.apparent_magnitude_limit-25-5*np.log10(
            self.fiducial_velocity / self.cosmology.H0)
        return abs_mag_lim

    def shecter_function(self, magnitdues):
        """Shecter Luminosity Function as proposed by Kochanek et al. (2001)."""
        constant = 0.4 * np.log(10) * self.phi_star
        term_1 = 10**(0.4 * (self.alpha+1) * (self.m_star-magnitdues))
        term_2 = np.exp(-10**(0.4 * (self.m_star-magnitdues)))
        return constant * term_1 * term_2
    
    def M_12(self, v_avg):
        """Works out average magnitude."""
        return self.apparent_magnitude_limit - 25 - 5*np.log10(v_avg/self.cosmology.H0)
