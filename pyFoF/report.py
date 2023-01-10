"""Report class with clustering metrics to be used for reporting."""
import pandas as pd
import numpy as np
from data_handling import read_data

import sklearn.metrics 

valid_metrics_dict = {
    'silhouette_score': sklearn.metrics.cluster.silhouette_score,
    'calinski_harabasz_score': sklearn.metrics.cluster.calinski_harabasz_score,
    'davies_bouldin_score': sklearn.metrics.cluster.davies_bouldin_score
}

class GroupReport:
    """
    Class to generate a metric report for a galaxy group output from a FoF algorithm or other algorithm.
    Returns neatly formatted metrics to be printed to console output or other output, and can return a dict
    with metric values for further processing or visualisation.

    Parameters
    ----------

    Returns
    -------
    report : str or dict
        Text summary of the reporting metrics (unsupervised) for the dataset provided.
        Dictionary returned if output_dict is True. Dictionary has the
        following structure::
            {'silhouette_score':0.5,
             'calinski_harabasz_score':1.0,
             'davies_bouldin_score':0.67,
            }
    """

    def __init__(self, dataset, reporting_metric_subset = 'all', output_dict = True):

        if type(dataset) != pd.DataFrame:
            raise TypeError('Type of dataset provided must be a Pandas Dataframe.')

        self.dataset = dataset

        self.group_labels = dataset['group_id']
        self.X = np.array(self.dataset.drop(['group_id'], axis = 1))

        if reporting_metric_subset != 'all':
            if type(reporting_metric_subset) != list:
                raise TypeError('Reporting metrics specified must be provided either as str or list of str containing names of metrics to run.')
            for metric_name in reporting_metric_subset:
                if metric_name not in valid_metrics_dict:
                    raise ValueError('Invalid reporting metric specified. Must be one of silhouette_score, calinski_harabasz_score, or davies_bouldin_score')

        self.reporting_metric_subset = reporting_metric_subset

        if self.reporting_metric_subset == 'all':
            self.reporting_metric_subset = ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']

        self.output_dict = output_dict
        self.metrics_dict = {}

    def _generate_metrics(self):

        metrics_dict = {}
        for metric_name in self.reporting_metric_subset:
            metrics_dict[metric_name] = valid_metrics_dict[metric_name](self.X, self.group_labels)

        self.metrics_dict = metrics_dict

    def generate(self):
        self._generate_metrics()

        width = max([len(metric_name) for metric_name in valid_metrics_dict.keys()])
        digits = 2

        row_fmt = "{:>{width}s} " + " {:>9.{digits}f}"
        for metric_name in self.metrics_dict.keys():
            print(row_fmt.format(metric_name, self.metrics_dict[metric_name], width=width, digits=digits))

        if self.output_dict:
            return self.metrics_dict
        else:
            return

if __name__ == "__main__":
    INFILE = '../data/Test_Data/group_catalog.fits'
    #INFILE = './data/Kids/WISE-SGP_redshifts_w1mags.tbl'
    #INFILE = './data/Test_Data/Test_Cat.tbl'
    data = read_data(INFILE)