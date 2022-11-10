"""Friends-of-friends core functions."""

import numpy as np
from .survey import Survey
from .utils import calculate_angular_seperation

class Trial:
    """Class of a single trial run."""
    def __init__(self, survey: Survey, d_0: float, v_0: float):
        self.survey = survey
        self.d_0 = d_0
        self.v_0 = v_0

    def find_friends_from_point(self, ra, dec, vel, data_frame):
        """Finds all the friends around a point (ra, dec, vel)"""
        velocity_df = data_frame[data_frame['vel'].between(vel - self.v0, vel + self.v0)]
        vels = np.array(velocity_df['vel'])
        ras = np.array(velocity_df['ra'])
        decs = np.array(velocity_df['dec'])

        separations = calculate_angular_seperation(ra, dec, ras, decs)
        theta = (np.pi/180) * (separations/2)
        v_average = (vel + vels)/2
        D_12 = np.sin(theta) * (v_average/self.survey.H0)
