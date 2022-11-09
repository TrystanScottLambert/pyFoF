"""Friends-of-friends core functions."""

import numpy as np

#Shectar parameters
#alpha, M_star, Phi_star = -1.02, -24.2, 0.0108
#lum_const = 0.4 * np.log(10) * Phi_star
#survey parameters
APPARENT_MAGNITUDE_LIMIT = 11.75
FIDUCIAL_VELOCITY = 1000.

def shecter_function(magnitdues, alpha, m_star, phi_star):
    """Shecter Luminosity Function as proposed by Kochanek et al. (2001)."""
    constant = 0.4 * np.log(10) * phi_star
    term_1 = 10**(0.4*(alpha+1)*(m_star-magnitdues))
    term_2 = np.exp(-10**(0.4*(m_star-magnitdues)))
    return constant * term_1 * term_2
