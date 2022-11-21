"""Friends-of-friends core functions."""

import numpy as np
from scipy.integrate import cumtrapz
from astropy.cosmology import FlatLambdaCDM
from survey import Survey
from utils import calculate_angular_seperation, wrap_mean
from data import read_data
from group import Group

class Trial:
    """Class of a single trial run."""
    def __init__(self, survey: Survey, d_0: float, v_0: float, d_max: float, v_max: float) -> None:
        """Initializing. Requires linking lengths d_0 and v_0 which are the on-sky linking length
        in Mpc and the line-of-sight linking length in km/s respectively. As well as the maximum
        on-sky radius of a group (d_max) in Mpc and the maximum line-of-sight distance (v_max)
        in kpc."""
        self.survey = survey
        self.d_0 = d_0
        self.v_0 = v_0
        self.v_max = v_max
        self.d_max = d_max
        self.right_ascension = self.survey.data_frame['ra'].values
        self.dec = self.survey.data_frame['dec'].values
        self.vel = self.survey.data_frame['vel'].values
        self.mag = self.survey.data_frame['W1'].values

    def _calculate_projected_separations(self, right_ascension, dec, vel, ras, decs, vels):
        """Works out the projected distances in Mpc (as apoosed to arcseconds)."""
        separations = calculate_angular_seperation(right_ascension, dec, ras, decs)
        theta = (np.pi/180) * (separations/2)
        v_averages = (vel + vels)/2
        on_sky_distances_mpc = np.sin(theta) * (v_averages/self.survey.cosmology.H0.value)
        return on_sky_distances_mpc

    def find_friends_from_point(self, right_ascension: float, dec: float, vel: float, free_galaxies: np.ndarray) -> np.ndarray:
        """Finds all the friends around a point (ra, dec, vel)"""
        available_data_frame = self.survey.data_frame.iloc[free_galaxies]
        velocity_df = available_data_frame[available_data_frame['vel'].between(vel - self.v_0, vel + self.v_0)]
        vels = velocity_df['vel'].values
        ras = velocity_df['ra'].values
        decs = velocity_df['dec'].values
        
        on_sky_distances_mpc = self._calculate_projected_separations(right_ascension, dec, vel, ras, decs, vels)
        v_averages = (vel + vels)/2
        upper_limits = self.survey.m_12(v_averages)
        upper_limits_sort = np.sort(upper_limits)
        arg = upper_limits.argsort()
        rev = arg.argsort()
        upper_limits_sort = np.append(np.array([-32]), upper_limits_sort)
        yts = self.survey.shecter_function(upper_limits_sort)
        numerator = cumtrapz(yts, upper_limits_sort)
        numerator = numerator[rev]

        d_limits  = self.d_0 * (numerator/self.survey.integral[0])**(-1./3)

        check_limits = np.where(d_limits>self.d_max)[0] # This is very hacky
        if len(check_limits) != 0:
            d_limits[check_limits] = self.d_max

        cut = np.where(on_sky_distances_mpc < d_limits)[0]
        ids = velocity_df.iloc[cut]['fof_ids'].values
        return ids

    def find_friends_of_galaxy(self, galaxy_index: int, free_galaxies: np.ndarray) -> np.ndarray:
        """Finds the friends of a galaxy."""
        vel = self.vel[galaxy_index]
        right_ascension = self.right_ascension[galaxy_index]
        dec = self.dec[galaxy_index]
        return self.find_friends_from_point(right_ascension, dec, vel, free_galaxies)

    def _remove_outlying_members(self, members: np.ndarray) -> np.ndarray:
        """Takes the maximum values a group can be into account and removes outlying galxies."""
        group_ra = wrap_mean(self.right_ascension[members])
        group_dec = np.mean(self.dec[members])
        group_vel = np.mean(self.vel[members])

        projected_separations = self._calculate_projected_separations(
            group_ra, group_dec, group_vel,self.right_ascension[members],
            self.dec[members], self.vel[members])
        line_of_sight_distances = np.abs(group_vel - self.vel[members])
        on_sky_cut = np.where(projected_separations <= self.d_max)[0]
        line_of_sight_cut = np.where(line_of_sight_distances <= self.v_max)[0]
        galaxies_in_max_limits = np.intersect1d(on_sky_cut, line_of_sight_cut)
        return members[galaxies_in_max_limits]

    def find_group(self, index, free_galaxies):
        """Will find the group starting from the indexed galaxy."""

        friends_after = self.find_friends_of_galaxy(index, free_galaxies)
        friends_before = np.array([])

        iterations = 0
        while np.array_equal(friends_after, friends_before) is False:
            iterations += 1
            friends_before = friends_after
            friends_after = np.unique(np.concatenate(
                [self.find_friends_of_galaxy(friend, free_galaxies) for friend in friends_before]))
            friends_after = self._remove_outlying_members(friends_after)
        return friends_after

    def run(self):
        """Run one friends-of-friends trial."""
        checked = np.zeros(len(self.survey.data_frame))
        galaxies_left = np.where(checked == 0)[0]
        groups = []
        while len(galaxies_left) > 0:
            new_group = self.find_group(np.random.choice(galaxies_left), galaxies_left)
            groups.append(Group(new_group))
            checked[new_group] = 1
            galaxies_left = np.where(checked == 0)[0]
            print(f'{round(100*(1 - len(galaxies_left)/len(checked)),2)} %')
        return groups


if __name__ == '__main__':
    INFILE = '/home/trystan/Desktop/Work/pyFoF/data/Kids/Kids_S_hemispec_no_dupes_updated.tbl'
    INFILE = '/home/trystan/Desktop/Work/pyFoF/data/Kids/WISE-SGP_redshifts_w1mags.tbl'
    #INFILE = '/home/trystan/Desktop/Work/pyFoF/data/Test_Data/Test_Cat.tbl'
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    data = read_data(INFILE)
    KIDS = Survey(data, cosmo, 11.75)
    KIDS.convert_z_into_cz('zcmb')
    test_run = Trial(KIDS, d_0=0.56, v_0=350., v_max = 1500., d_max = 2.0)
    test = test_run.run()
