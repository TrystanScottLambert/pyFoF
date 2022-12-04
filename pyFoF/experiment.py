"""Friends of friends group theory implemntation based on Lambert et, al. (2020)"""

import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from data_handling import read_data
from survey import Survey
from fof import Trial
from graph_theory import stabalize
from group import Group

columns_to_drop = (
    'members',
    'weights',
    'survey',
)

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
                member_data, self.survey, weights=group_theory_data[2]
                ) for member_data in group_theory_data[0]]
        group_dicts = [group.__dict__ for group in self.groups]
        self.group_df = pd.DataFrame(group_dicts)
        self.group_df.insert(0, 'group_id', np.arange(len(self.group_df)))

        self._add_group_info_to_df()
        self._drop_unnecessary_columns_from_group_df()
        self.group_table = self._convert_pd_to_fitstable(self.group_df)
        self.galaxy_table = self._convert_pd_to_fitstable(self.survey.data_frame)
        self.edge_data = group_theory_data[1]

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

    def _add_group_info_to_df(self):
        """Adds the group ID and the galaxy weights to the survey data frame."""
        #weight = np.ones(len(self.survey.data_frame)) * -1
        groups_ids = np.ones(len(self.survey.data_frame)) * -1
        for i, group_members in enumerate(self.group_df['members']):
            groups_ids[np.array(group_members)] = i
            #weight[np.array(group_members)] = np.array(self.group_df['weights'][i])
        #self.survey.data_frame['weight'] = weight
        self.survey.data_frame['group_id'] = groups_ids

    def _drop_unnecessary_columns_from_group_df(self):
        """Dropping unrequired columns in the group data frame"""
        for column in columns_to_drop:
            self.group_df.drop(column, axis = 1, inplace=True)

    @staticmethod
    def _convert_pd_to_fitstable(data_frame: pd.DataFrame) -> Table:
        """Takes a pandas dataframe and returns a fits table."""
        fits_table = Table.from_pandas(data_frame)
        return fits_table

    def write_group_catalog(self, outfile_name: str, overwrite = False) -> None:
        """Generates a group catalog as a fits file."""
        self.group_table.write(outfile_name, overwrite = overwrite)

    def write_galaxy_catalog(self, outfile_name: str, overwrite = False) -> None:
        """Generates a galaxy catalog as a fits file"""
        self.galaxy_table.write(outfile_name, overwrite = overwrite)

    def write_all_catalogs(self, overwrite = False) -> None:
        """Writes all the catalogs from the experiment."""
        self.write_group_catalog('group_catalog.fits', overwrite)
        self.write_galaxy_catalog('galaxy_catalog.fits', overwrite)

if __name__ == '__main__':
    INFILE = '/home/trystan/Desktop/Work/pyFoF/data/Kids/Kids_S_hemispec_no_dupes_updated.tbl'
    #INFILE = '/home/trystan/Desktop/Work/pyFoF/data/Kids/WISE-SGP_redshifts_w1mags.tbl'
    #INFILE = '/home/trystan/Desktop/Work/pyFoF/data/Test_Data/Test_Cat.tbl'
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    data = read_data(INFILE)
    KIDS = Survey(data, cosmo, 18.)
    KIDS.convert_z_into_cz('z_helio')
    #KIDS.make_mag_colum('W1')
    KIDS.data_frame['mag'] = np.random.normal(15, 2, len(KIDS.data_frame))
    test_run = Experiment(
        d0_initial=0.3, d0_final=0.8,
        v0_initial=100, v0_final=500,
        d_max=2., v_max=1000,
        n_trials=10, cutoff=0.5, survey = KIDS
        )
    test_run.write_all_catalogs(overwrite = True)
