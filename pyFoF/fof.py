"""Friends-of-friends core functions."""

import numpy as np
from scipy.integrate import quad
from survey import Survey
from utils import calculate_angular_seperation
from data import read_data
from astropy.cosmology import FlatLambdaCDM

class Trial:
    """Class of a single trial run."""
    def __init__(self, survey: Survey, d_0: float, v_0: float):
        self.survey = survey
        self.d_0 = d_0
        self.v_0 = v_0

    def find_friends_from_point(self, ra, dec, vel, data_frame):
        """Finds all the friends around a point (ra, dec, vel)"""
        velocity_df = data_frame[data_frame['vel'].between(vel - self.v0, vel + self.v0)]
        vels = np.array(list(velocity_df['vel']))
        ras = np.array(list(velocity_df['ra']))
        decs = np.array(list(velocity_df['dec']))

        separations = calculate_angular_seperation(ra, dec, ras, decs)
        theta = (np.pi/180) * (separations/2)
        v_averages = (vel + vels)/2
        on_sky_distances_mpc = np.sin(theta) * (v_averages/self.survey.H0)

        upper_limits = self.survey.m_12(v_averages)
        numerator = quad(self.survey.shecter_function, -np.inf, upper_limits)
        d_limits  = self.d_0 * (numerator/self.survey.integral)**(-1./3)
        cut = np.where(on_sky_distances_mpc < d_limits)[0]
        return cut

if __name__ == '__main__':
    INFILE = '/home/trystan/Desktop/Work/pyFoF/data/Kids/Kids_S_hemispec_no_dupes_updated.tbl'
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    data_frame = read_data(INFILE)
    KIDS = Survey(data_frame, cosmo, 11.75)
