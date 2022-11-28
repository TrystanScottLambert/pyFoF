"""Friends of friends group theory implemntation based on Lambert et, al. (2020)"""

import numpy as np
from astropy.cosmology import FlatLambdaCDM
from data_handling import read_data
from survey import Survey
from fof import Trial
from group_theory import stabalize
from group import Group

class Experiment:
    """Class for one run of the modern algorithm."""
    def __init__(
        self, d0_initial, d0_final, v0_initial, v0_final,
        d_max, v_max, n_trials, cutoff, survey):
        """Initializing."""
        self.d0s = np.linspace(d0_initial, d0_final, n_trials)
        self.v0s = np.linspace(v0_initial, v0_final, n_trials)
        self.v_max = v_max
        self.d_max = d_max
        self.survey = survey
        self.number_of_trials = n_trials
        members = self.run()
        group_theory_data = stabalize(members, cutoff, n_trials)
        self.groups = [
            Group(
                member_data, self.survey, weights=group_theory_data[3]
                ) for member_data in group_theory_data[0]]
        #self.stable_arrays  = group_theory_data[0]
        self.edge_data = group_theory_data[1]
        #self.weights = group_theory_data[2]
        #self.weights_normed = group_theory_data[3]

    def run(self):
        """Runs the algorithm."""
        results = [
            Trial(
                self.survey, {
                    "d_0": self.d0s[i], "v_0": self.v0s[i], "v_max": self.v_max, "d_max": self.d_max
                    }
                ).run() for i in range(self.number_of_trials)
            ]
        concatenated_results = np.concatenate(results)
        members_list = [group.members for group in concatenated_results]
        return members_list


if __name__ == '__main__':
    INFILE = '/home/trystan/Desktop/Work/pyFoF/data/Kids/Kids_S_hemispec_no_dupes_updated.tbl'
    INFILE = '/home/trystan/Desktop/Work/pyFoF/data/Kids/WISE-SGP_redshifts_w1mags.tbl'
    INFILE = '/home/trystan/Desktop/Work/pyFoF/data/Test_Data/Test_Cat.tbl'
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    data = read_data(INFILE)
    KIDS = Survey(data, cosmo, 11.75)
    KIDS.convert_z_into_cz('zcmb')
    KIDS.make_mag_colum('W1')
    test_run = Experiment(
        d0_initial=0.3, d0_final=0.6,
        v0_initial=100, v0_final=400,
        d_max=2., v_max=1000,
        n_trials=3, cutoff=0.5, survey = KIDS
        )
