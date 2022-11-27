"""Friends of friends group theory implemntation based on Lambert et, al. (2020)"""

import numpy as np
from astropy.cosmology import FlatLambdaCDM
from data import read_data
from survey import Survey
from fof import Trial
from group_theory import stabalize

class Run:
    """Class for one run of the modern algorithm."""
    def __init__(
        self, d0_initial, d0_final, v0_initial, v0_final,
        d_max, v_max, n_trials, survey):
        """Initializing."""
        self.d0s = np.linspace(d0_initial, d0_final, n_trials)
        self.v0s = np.linspace(v0_initial, v0_final, n_trials)
        self.v_max = v_max
        self.d_max = d_max
        self.survey = survey
        self.number_of_trials = n_trials

    def run(self):
        """Runs the algorithm."""
        results = [
            Trial(
                self.survey, {
                    "d_0": self.d0s[i], "v_0": self.v0s[i], "v_max": self.v_max, "d_max": self.d_max
                    }
                ).run() for i in range(self.number_of_trials)
            ]
        return results


if __name__ == '__main__':
    INFILE = '/home/trystan/Desktop/Work/pyFoF/data/Kids/Kids_S_hemispec_no_dupes_updated.tbl'
    INFILE = '/home/trystan/Desktop/Work/pyFoF/data/Kids/WISE-SGP_redshifts_w1mags.tbl'
    #INFILE = '/home/trystan/Desktop/Work/pyFoF/data/Test_Data/Test_Cat.tbl'
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    data = read_data(INFILE)
    KIDS = Survey(data, cosmo, 11.75)
    KIDS.convert_z_into_cz('zcmb')
    test_run = Run(0.3, 0.6, 300, 400, 2., 1000., 3, KIDS)
    groups = test_run.run()
    groups = np.concatenate(groups)
    run_result = [group.members for group in groups]
    stabalize(run_result, 0.5, 3)
