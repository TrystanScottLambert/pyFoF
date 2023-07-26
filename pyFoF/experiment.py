"""Friends of friends group theory implemntation based on Lambert et, al. (2020)"""
import multiprocessing

from concurrent.futures import ProcessPoolExecutor
from typing import Optional
import rich.progress as rp

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
        self, d0_initial: float, d0_final: float, v0_initial: float, v0_final: float,
        d_max: float, v_max: float, n_trials: int, cutoff: float, survey: Survey,
        n_workers: Optional[int] = None):
        """Initializing."""
        self.d0s = np.linspace(d0_initial, d0_final, n_trials)
        self.v0s = np.linspace(v0_initial, v0_final, n_trials)
        self.v_max = v_max
        self.d_max = d_max
        self.survey = survey
        self.number_of_trials = n_trials
        if n_workers:
            self.n_workers = n_workers
        else:
            self.n_workers = multiprocessing.cpu_count()

        self.cutoff = cutoff

    def run(self, 
            use_multiprocessing: bool = True):
        """
        Runs the algorithm.
        
        Parameters
        ----------
        use_multiprocessing : bool
            Whether to use multiprocessing for parallelisation of experiment trial runs. Default is true.
        
        Returns
        -------
            self.members : np.array
                The list of members and associated galaxy group IDs to each member of a group
        """
        if use_multiprocessing:
            try:
                # to ensure forwards and backwards compatibility with various OS versions which use spawn rather than fork 
                # as the default process creation method. Spawn creates thread-related Runtime Errors related to children spawning 
                # faster than parent processes can finish spawning, and require tighter controls not necessary for these use-cases.
                multiprocessing.set_start_method('fork') 
            except RuntimeError as runtime_error:
                # Just print(e) is cleaner and more likely what we want,
                # but if lets insist on printing message specifically whenever possible...
                if hasattr(runtime_error, 'message'):
                    error_message = runtime_error.message
                else:
                    error_message = runtime_error
                
                if str(error_message) == 'context has already been set':
                    print('WARN: Multiprocessing context has already been set and is trying to be reset. You can only set this context once per Python session. Continuing with default')
                else:
                    print('ERROR: Multiprocessing has thrown an error on setting context to "fork". Traceback raised is printed below. Recommended setting is to run with use_multiprocessing=False and raise an issue in the GitHub repository for pyFoF.')
                    raise
            
            with rp.Progress(
                "[progress.description]{task.description}",
                rp.BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                rp.TimeRemainingColumn(),
                rp.TimeElapsedColumn(),
                refresh_per_second=1,  # bit slower updates
            ) as progress:
                futures = []  # keep track of the jobs
                with multiprocessing.Manager() as manager:
                    # this is the key - we share some state between our
                    # main process and our worker functions
                    _progress = manager.dict()
                    overall_progress_task = progress.add_task("[green]All jobs progress:")

                    with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                        for n in range(0, self.number_of_trials):  # iterate over the jobs we need to run
                            # set visible false so we don't have a lot of bars all at once:
                            task_id = progress.add_task(f"Trial {n+1} (d_0 = {np.round(self.d0s[n], 2)}, v_0 = {np.round(self.v0s[n], 2)}): ", visible=False)
                            futures.append(executor.submit( Trial(
                                                            self.survey, {
                                                                "d_0": self.d0s[n], "v_0": self.v0s[n], "v_max": self.v_max, "d_max": self.d_max
                                                                }
                                                            ).run,
                                                            progress_mode='experiment',
                                                            progress_bar=_progress, 
                                                            task_id=task_id))

                        # monitor the progress:
                        while (n_finished := sum([future.done() for future in futures])) < len(
                            futures
                        ):
                            progress.update(
                                overall_progress_task, completed=n_finished, total=len(futures)
                            )
                            for task_id, update_data in _progress.items():
                                latest = update_data["progress"]
                                total = update_data["total"]
                                # update the progress bar for this task:
                                progress.update(
                                    task_id,
                                    completed=latest,
                                    total=total,
                                    visible=latest < total,
                                )

                        # raise any errors
                        results = []
                        for future in futures:
                            results.append(future.result())
                        concatenated_results = np.concatenate(results)
                        members_list = [group.members for group in concatenated_results]

            #assert len(results) == self.number_of_trials
            self.members = members_list
        else:
            results = [
                Trial(
                    self.survey, {
                        "d_0": self.d0s[i], "v_0": self.v0s[i], "v_max": self.v_max, "d_max": self.d_max
                        }
                    ).run(progress_mode='none') for i in range(self.number_of_trials)
                ]
            concatenated_results = np.concatenate(results)
            members_list = [group.members for group in concatenated_results]
            self.members = members_list

        # run stabilisation, i.e. graph theory post-processing
        graph_theory_post_processed_output = stabalize(self.members, self.cutoff, self.number_of_trials)

        # retrieve groups
        self.groups = [
            Group(
                member_data, self.survey, weights=graph_theory_post_processed_output[2]
                ) for member_data in graph_theory_post_processed_output[0]]
        group_dicts = [group.__dict__ for group in self.groups]
        # transform groups into Pandas DataFrame with group IDs as group_id
        self.group_df = pd.DataFrame(group_dicts)
        self.group_df.insert(0, 'group_id', np.arange(len(self.group_df)))

        # clean up operations for above dataframe (not passed as parameter, used as class var)
        self._add_group_info_to_df()
        self._drop_unnecessary_columns_from_group_df()

        # transform as FITS table to be exported for other secondary use-cases by user
        self.group_table = self._convert_pd_to_fitstable(self.group_df)
        self.galaxy_table = self._convert_pd_to_fitstable(self.survey.data_frame)
        self.edge_data = graph_theory_post_processed_output[1]

        return self.members

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
        """
        Takes a pandas dataframe and returns a fits table.
        """
        fits_table = Table.from_pandas(data_frame)
        return fits_table

    def get_survey_input_data(self):
        """
        Function to return the survey input data passed into the experiment
        """
        return self.survey.data_frame

    def get_groups(self, output_type='pandas'):
        """
        Function to return the galaxy groups calculated by the FoF algorithm and experiments

        Parameters
        ----------
        output_type : str
            The output type to return, the options here are 'pandas' for a Pandas DataFrame or
            'fits' for a FITS-friendly table.
        
        Returns
        -------

        """
        if output_type not in ['pandas', 'fits']:
            raise ValueError('Incorrect value specified for output_type argument. Accepted values are "pandas" and "fits".')
        if output_type == 'pandas':
            return self.group_df
        elif output_type == 'fits':
            return self.group_table

    def get_edge_data(self):
        """
        Function to return the edge data calculated after the graph theory post-processing
        """
        return self.edge_data
    
    def write_edge_data(self) -> None:
        """Writes the edge data as a txt file for edge visualizaton."""
        f = open('edge_data.txt', 'w')
        f.write('# ID1, ID2, WEIGHT')
        edge_data = self.get_edge_data()
        for edge_datum in edge_data:
            f.write(f'{int(edge_datum[0])} {int(edge_datum[1])} {edge_datum[2]} \n')
        f.close()
        
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
    INFILE = '../data/Kids/Kids_S_hemispec_no_dupes_updated.tbl'

    #INFILE = './data/Kids/WISE-SGP_redshifts_w1mags.tbl'
    #INFILE = './data/Test_Data/Test_Cat.tbl'
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    data = read_data(INFILE)
    KIDS = Survey(data, cosmo, 18.)
    KIDS.convert_z_into_cz('z_helio')
    KIDS.add_positional_information_to_df('vel')
    #KIDS.make_mag_colum('W1')
    KIDS.data_frame['mag'] = np.random.normal(15, 2, len(KIDS.data_frame))
    test_run = Experiment(
        d0_initial=0.3, d0_final=0.8,
        v0_initial=100, v0_final=500,
        d_max=2., v_max=1000,
        n_trials=10, cutoff=0.5, survey = KIDS
        )
    test_run.run()
    test_run.write_all_catalogs(overwrite = True)
    test_run.write_edge_data()
