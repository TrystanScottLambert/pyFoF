"""Friends-of-friends core functions."""

import numpy as np
from scipy.integrate import cumtrapz
from astropy.cosmology import FlatLambdaCDM
from survey import Survey
from utils import calculate_angular_seperation, wrap_mean
from data import read_data
import pylab as plt


class Trial:
    """Class of a single trial run."""
    def __init__(self, survey: Survey, d_0: float, v_0: float):
        self.survey = survey
        self.d_0 = d_0
        self.v_0 = v_0

    def find_friends_from_point(self, ra, dec, vel):
        """Finds all the friends around a point (ra, dec, vel)"""
        data_frame = self.survey.data_frame
        velocity_df = data_frame[data_frame['vel'].between(vel - self.v_0, vel + self.v_0)]
        vels = velocity_df['vel'].values
        ras = velocity_df['ra'].values
        decs = velocity_df['dec'].values

        separations = calculate_angular_seperation(ra, dec, ras, decs)
        theta = (np.pi/180) * (separations/2)
        v_averages = (vel + vels)/2
        on_sky_distances_mpc = np.sin(theta) * (v_averages/self.survey.cosmology.H0.value)

        upper_limits = self.survey.m_12(v_averages)
        upper_limits_sort = np.sort(upper_limits)
        arg = upper_limits.argsort()
        rev = arg.argsort()
        upper_limits_sort = np.append(np.array([-32]), upper_limits_sort)
        yts = self.survey.shecter_function(upper_limits_sort)
        numerator = cumtrapz(yts, upper_limits_sort)
        numerator = numerator[rev]

        d_limits  = self.d_0 * (numerator/self.survey.integral[0])**(-1./3)
        cut = np.where(on_sky_distances_mpc < d_limits)[0]
        return cut
    
    def find_group(self, index):
        """Will find the group starting from the indexed galaxy."""
        ra = self.survey.data_frame['ra'].values
        dec = self.survey.data_frame['dec'].values
        vel = self.survey.data_frame['vel'].values

        friends_after = self.find_friends_from_point(ra[index], dec[index], vel[index])

        friends_before = np.array([])
        iterations = 1
        while friends_after != friends_before:
            iterations += 1
            print(iterations)
            friends_before = friends_after
            print(friends_before)
            group_ra = wrap_mean(ra[friends_before])
            group_dec = np.mean(dec[friends_before])
            group_vel = np.mean(vel[friends_before])
            fig = plt.figure()
            ax = fig.add_subplot(projection='aitoff')
            ax.scatter(group_ra, group_dec, marker = 's')
            ax.scatter(ra[friends_before], dec[friends_before], color='r')
            ax.scatter(ra, dec, s=0.1, color='k')
            plt.show()

            friends_after = self.find_friends_from_point(group_ra, group_dec, group_vel)
            print(friends_after)
            print()
        return friends_after


if __name__ == '__main__':
    INFILE = '/home/trystan/Desktop/Work/pyFoF/data/Kids/Kids_S_hemispec_no_dupes_updated.tbl'
    INFILE = '/home/trystan/Desktop/Work/pyFoF/data/Kids/WISE-SGP_redshifts_w1mags.fits'
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    data = read_data(INFILE)
    KIDS = Survey(data, cosmo, 11.75)
    KIDS.convert_z_into_cz('zcmb')
    test_run = Trial(KIDS, d_0=0.56, v_0=350.)
    #test = test_run.find_friends_from_point(3.62163, -33.11417, 9080.713594990893)
    test = test_run.find_group(1)
