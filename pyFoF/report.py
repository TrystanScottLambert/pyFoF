"""Report class with clustering metrics to be used for reporting."""
import pandas as pd
import numpy as np
from .data_handling import read_data
from astropy.cosmology import FlatLambdaCDM
from .survey import Survey

from .metrics import redshift_silhouette_score, redshift_calinski_harabasz_score, redshift_davies_bouldin_score, redshift_dunn_index
from .utils import redshift_projected_unscaled_separation

import sklearn.metrics
import uuid

valid_metrics_dict = {
    'silhouette_score': redshift_silhouette_score,
    'calinski_harabasz_score': redshift_calinski_harabasz_score,
    'davies_bouldin_score': redshift_davies_bouldin_score,
    'dunn_index_score': redshift_dunn_index
}

valid_sklearn_metrics_dict = {
    'adjusted_mutual_info_score': sklearn.metrics.adjusted_mutual_info_score,
    'adjusted_rand_score': sklearn.metrics.adjusted_rand_score,
    'completeness_score': sklearn.metrics.completeness_score,
    'contingency_matrix': sklearn.metrics.cluster.contingency_matrix,
    'pair_confusion_matrix': sklearn.metrics.cluster.pair_confusion_matrix,
    'fowlkes_mallows_score': sklearn.metrics.fowlkes_mallows_score,
    'homogeneity_score': sklearn.metrics.homogeneity_score,
    'v_measure_score': sklearn.metrics.v_measure_score
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

    def __init__(self,
                dataset: pd.DataFrame,
                h0_value: float,
                reporting_metric_subset: str = 'all',
                output_dict: bool = True,
                column_names: list[str] = ['ra', 'dec', 'vel'],
                group_id_column_name: str = 'group_id',
                silhouette_subset_frac: float = 1.0):

        if reporting_metric_subset != 'all':
            if type(reporting_metric_subset) != list:
                raise TypeError('Reporting metrics specified must be provided either as str or list of str containing names of metrics to run.')
            for metric_name in reporting_metric_subset:
                if metric_name not in valid_metrics_dict:
                    raise ValueError('Invalid reporting metric specified. Must be one of silhouette_score, calinski_harabasz_score, or davies_bouldin_score')

        if silhouette_subset_frac <= 0.0 or silhouette_subset_frac > 1.0:
            raise ValueError(f'Values for silhouette_subset_frac must be between 0 and 1 of type float. Value provided is {silhouette_subset_frac}')
        
        self.silhouette_subset_frac = silhouette_subset_frac
        self.reporting_metric_subset = reporting_metric_subset

        if self.reporting_metric_subset == 'all':
            self.reporting_metric_subset = ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']

        if not group_id_column_name in dataset.columns:
            raise ValueError(f'Column name provided for group IDs in parameter group_id_column_name is not present in dataset columns! Value provided was {group_id_column_name}. Please use one of dataset.column values to specify the field name.')
        
        for column_name in column_names:
            if not column_name in dataset.columns:
                raise ValueError(f'One column name provided in parameter column_names is not present in dataset columns! Value provided was {column_name}. Please use one of dataset.column values to specify the field name.')

        self.column_names = column_names
        self.group_id_column_name = group_id_column_name
        self.h0_value = h0_value

        self.dataset = dataset[self.column_names]

        self.X = np.array(dataset[self.column_names])
        self.group_labels = dataset[self.group_id_column_name]

        self.X_frac = dataset[self.column_names].sample(frac = self.silhouette_subset_frac)
        self.group_labels_frac = self.group_labels[self.X_frac.index]

        self.output_dict = output_dict
        self.metrics_dict = {}

    def _generate_metrics(self):

        metrics_dict = {}
        for metric_name in self.reporting_metric_subset:
            metric_func = valid_metrics_dict[metric_name]

            if ('silhouette' in metric_name) or ('dunn' in metric_name):
                X_eval = self.X_frac.copy()
                group_labels_eval = self.group_labels_frac.copy()
            else:
                X_eval = self.X.copy()
                group_labels_eval = self.group_labels.copy()

            if 'silhouette' in metric_name:
                metrics_dict[metric_name] = metric_func(X_eval,
                                                        group_labels_eval,
                                                        metric = redshift_projected_unscaled_separation, 
                                                        n_jobs=-1)
            else:
                metrics_dict[metric_name] = metric_func(X_eval,
                                                        group_labels_eval,
                                                        h0_value = self.h0_value,
                                                        column_names = self.column_names, 
                                                        wrap_columns = ['ra'],
                                                        metric = redshift_projected_unscaled_separation, 
                                                        n_jobs=-1)    

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


class GroupComparisonReport:
    """
    Class to generate metric reports for a list galaxy group outputs from a FoF algorithm(s) or other algorithm,
    and to compare these reports via unsupervised and supervised comparison metrics via scikit-learn and custom implementations.

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

    def __init__(self,
                datasets: list[pd.DataFrame],
                h0_value: float,
                unsupervised_metric_subset: str = 'all',
                supervised_metric_subset: str = 'all',
                generate_contingency: bool = False,
                generate_confusion_matrices: bool = False,
                output_dict: bool = True,
                dataset_names: list[str] = None,
                column_names: list[str] = ['ra', 'dec', 'vel'],
                group_id_column_name: str = 'group_id',
                silhouette_subset_frac: float = 1.0):

        if unsupervised_metric_subset != 'all':
            if type(unsupervised_metric_subset) != list:
                raise TypeError('Reporting metrics specified must be provided either as str or list of str containing names of metrics to run.')
            for metric_name in unsupervised_metric_subset:
                if metric_name not in valid_metrics_dict:
                    raise ValueError('Invalid reporting metric specified. Must be one of silhouette_score, calinski_harabasz_score, or davies_bouldin_score')

        if supervised_metric_subset != 'all':
            if type(supervised_metric_subset) != list:
                raise TypeError('Reporting metrics specified must be provided either as str or list of str containing names of metrics to run.')
            for metric_name in supervised_metric_subset:
                if metric_name not in valid_metrics_dict:
                    raise ValueError('Invalid reporting metric specified. Must be one of silhouette_score, calinski_harabasz_score, or davies_bouldin_score')

        if silhouette_subset_frac <= 0.0 or silhouette_subset_frac > 1.0:
            raise ValueError(f'Values for silhouette_subset_frac must be between 0 and 1 of type float. Value provided is {silhouette_subset_frac}')
        
        self.silhouette_subset_frac = silhouette_subset_frac
        self.unsupervised_metric_subset = unsupervised_metric_subset
        self.supervised_metric_subset = supervised_metric_subset

        if self.unsupervised_metric_subset == 'all':
            self.unsupervised_metric_subset = ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']

        if self.supervised_metric_subset == 'all':
            self.supervised_metric_subset = ['adjusted_mutual_info_score', 'adjusted_rand_score', 'completeness_score', 'fowlkes_mallows_score', 'homogeneity_score', 'v_measure_score']

        for dataset in datasets:
            if not group_id_column_name in dataset.columns:
                raise ValueError(f'Column name provided for group IDs in parameter group_id_column_name is not present in at least one of the provided dataset columns! Value provided was {group_id_column_name}. Please use one of dataset.column values to specify the field name.')
        
        if (type(generate_contingency) != bool) or (type(generate_confusion_matrices) != bool):
            raise TypeError('Types of arguments generate_contingency and generate_confusion_matrices must be bool! Please use True or False as inputs.')

        self.generate_contingency = generate_contingency
        self.generate_confusion_matrices = generate_confusion_matrices

        for column_name in column_names:
            dataset_num = 0
            for dataset in datasets:
                dataset_num += 1
                if not column_name in dataset.columns:
                    raise ValueError(f'One column name provided in parameter column_names is not present in at least one of the provided dataset columns! Value provided was {column_name}. Dataset checked was #{dataset_num} in the list. Please use one of dataset.column values to specify the field name.')

        if type(datasets) != list:
            raise TypeError('Type of argument datasets is incorrect. Expected type to be list of Pandas DataFrames.')

        dataset_num = 0
        for dataset in datasets:
            dataset_num += 1
            if type(dataset) != pd.DataFrame:
                raise TypeError(f'Type of argument of a dataset is incorrect. Expected type of dataframe #{dataset_num} to be Pandas DataFrames. Please convert it.')

        if dataset_names is not None:
            if type(dataset_names) != list:
                raise TypeError(f'Type of argument dataset_names is not list. Expected a list of names for each specified dataset, {len(datasets)} in total.')
            if len(dataset_names) != len(datasets):
                raise ValueError(f'Number of dataset names doesn\'t match number of datasets provided. Expected a list of names for each specified dataset, {len(datasets)} in total.')
            for dataset_name in dataset_names:
                if type(dataset_name) != str:
                    raise TypeError('Type of each dataset name must be of type str.')
        
        self.column_names = column_names
        self.group_id_column_name = group_id_column_name
        self.h0_value = h0_value

        if dataset_names is not None:
            self.dataset_names = dataset_names
        
        # subsetting columns for each dataframe provided
        self.datasets_metainfo = []
        dataset_id = 0
        for dataset in datasets:
            dataset_metainfo = {
                'dataset_id': dataset_id + 1,
                'dataset': dataset[self.column_names],
                'X': np.array(dataset[self.column_names]),
                'group_labels': dataset[self.group_id_column_name],
                'X_frac': dataset[self.column_names].sample(frac = self.silhouette_subset_frac),
                'group_labels_frac': dataset[self.group_id_column_name][self.X_frac.index]

            }

            if dataset_names is not None:
                dataset_metainfo['dataset_name'] = self.dataset_names[dataset_id]
            else:
                dataset_metainfo['dataset_name'] = uuid.uuid4()

            self.datasets_metainfo.append(dataset_metainfo)
            dataset_id += 1


        self.output_dict = output_dict
        self.metrics_dict = {}

    def _generate_supervised_metrics(self):

        matrix_dict = {}
        for metric_name in self.supervised_metric_subset:

            # NB: Currently not supporting contingency matrices due to their large memory needs for these large datasets
            if metric_name == 'contingency_matrix':
                continue
            
            if metric_name == 'pairwise_confusion_matrix' and self.generate_confusion_matrices:
                metric_func = valid_sklearn_metrics_dict[metric_name]   

                metric_matrix = [[(metric_func(dataset_metainfo_1['group_labels'], dataset_metainfo_2['group_labels']) if int(dataset_metainfo_1['dataset_id'])<int(dataset_metainfo_2['dataset_id']) else 0) for dataset_metainfo_2 in self.datasets_metainfo] for dataset_metainfo_1 in self.datasets_metainfo]
                matrix_dict[metric_name] = metric_matrix

                for i in range(len(self.datasets_metainfo)):
                    for j in range(len(self.datasets_metainfo)):
                        if i < j:
                            print("\t".join([self.datasets_metainfo[i]['dataset_name']]+[self.datasets_metainfo[j]['dataset_name']]))
                            print("\t".join([metric_matrix[i][j][0]]))
                            print("\t".join([metric_matrix[i][j][1]]))
                                            
            else:
                metric_func = valid_sklearn_metrics_dict[metric_name]   

                metric_matrix = [[(metric_func(dataset_metainfo_1['group_labels'], dataset_metainfo_2['group_labels']) if int(dataset_metainfo_1['dataset_id'])<int(dataset_metainfo_2['dataset_id']) else 0) for dataset_metainfo_2 in self.datasets_metainfo] for dataset_metainfo_1 in self.datasets_metainfo]
                matrix_dict[metric_name] = metric_matrix

                print("\t".join(['']+[dataset_metainfo['dataset_name'] for dataset_metainfo in self.datasets_metainfo]))
                for i in range(len(self.datasets_metainfo)):
                    print("\t".join([self.datasets_metainfo[i]['dataset_name']] + metric_matrix[i]))

        return matrix_dict
    
    def _generate_single_unsupervised_metric(self, dataset_metainfo):

        metrics_dict = {}
        for metric_name in self.unsupervised_metric_subset:
            metric_func = valid_metrics_dict[metric_name]

            if ('silhouette' in metric_name) or ('dunn' in metric_name):
                X_eval = dataset_metainfo['X_frac'].copy()
                group_labels_eval = dataset_metainfo['groups_labels_frac'].copy()
            else:
                X_eval = dataset_metainfo['X'].copy()
                group_labels_eval = dataset_metainfo['groups_labels'].copy()

            if 'silhouette' in metric_name:
                metrics_dict[metric_name] = metric_func(X_eval,
                                                        group_labels_eval,
                                                        metric = redshift_projected_unscaled_separation, 
                                                        n_jobs=-1)
            else:
                metrics_dict[metric_name] = metric_func(X_eval,
                                                        group_labels_eval,
                                                        h0_value = self.h0_value,
                                                        column_names = self.column_names, 
                                                        wrap_columns = ['ra'],
                                                        metric = redshift_projected_unscaled_separation, 
                                                        n_jobs=-1)    

        return metrics_dict

    def _generate_unsupervised_metrics(self):
        unsupervised_metrics_df = pd.DataFrame()
        for dataset_metainfo in self.datasets_metainfo:
            dataset_unsupervised_metrics_dict = self._generate_single_unsupervised_metric(dataset_metainfo=dataset_metainfo)
            dataset_unsupervised_metrics_dict['dataset_id'] = dataset_metainfo['dataset_id']
            dataset_unsupervised_metrics_dict['dataset_name'] = dataset_metainfo['dataset_name']

            unsupervised_metrics_df = unsupervised_metrics_df.append(dataset_unsupervised_metrics_dict, ignore_index=True)
          
        metrics_formatters = {}
        for colname in unsupervised_metrics_df.columns:
            if colname in ['dataset_id', 'dataset_name'] or unsupervised_metrics_df[colname].dtype == str or unsupervised_metrics_df[colname].dtype == object:
                metrics_formatters[colname] = "{:>{width}s} ".format
            else:
                metrics_formatters[colname] = " {:>9.{digits}f}".format

        unsupervised_metrics_str = unsupervised_metrics_df.to_string(formatters=metrics_formatters)
        print(unsupervised_metrics_str)

        return unsupervised_metrics_df

    def generate(self):

        unsupervised_metrics_df = self._generate_unsupervised_metrics()
        self.unsupervised_metrics_df = unsupervised_metrics_df

        self.metrics_dict['unsupervised_metrics'] = unsupervised_metrics_df

        supervised_metrics_df = self._generate_supervised_metrics()
        self.supervised_metrics_df = supervised_metrics_df

        self.metrics_dict['supervised_metrics'] = supervised_metrics_df

        if self.output_dict:
            return self.metrics_dict
        else:
            return

if __name__ == "__main__":
    GROUPS_INFILE = './group_catalog.fits'
    DATA_INFILE = read_data('../data/Kids/Kids_S_hemispec_no_dupes_updated.tbl')
    #INFILE = './data/Kids/WISE-SGP_redshifts_w1mags.tbl'
    #INFILE = './data/Test_Data/Test_Cat.tbl'
    datasurvey = read_data(DATA_INFILE)
    data = read_data('galaxy_catalog.fits')
    X = data[['ra', 'dec', 'vel']]
    labels_test = data['group_id']

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    KIDS = Survey(datasurvey, cosmo, 18.0)
    KIDS.convert_z_into_cz('z_helio')
    KIDS.data_frame['mag'] = np.random.normal(15, 2, len(KIDS.data_frame))
    H0_value = KIDS.cosmology.H0.value

    redshift_davies_bouldin_score(X, labels_test, H0_value, column_names=['ra', 'dec', 'vel'])
    