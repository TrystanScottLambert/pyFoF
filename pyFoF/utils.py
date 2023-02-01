"""Utilities which are needed for the package."""

import numpy as np
import pandas as pd 

from typing import Union

def calculate_angular_seperation(long_a, lat_a, long_b, lat_b):
    """
    Determines the angular separation between two points, or two catalogs.
    The inputs must be in degrees and can be floats or numpy arrays of floats.
    This is using the same formula as astropy. Under computational formulaes:
    https://en.wikipedia.org/wiki/Great-circle_distance
    """

    faq = (np.pi/180)
    long_a, long_b, lat_a, lat_b = long_a*faq, long_b*faq, lat_a*faq, lat_b*faq
    sin_difference_long = np.sin(long_b - long_a)
    cos_difference_long = np.cos(long_b - long_a)
    sin_lat_a = np.sin(lat_a)
    sin_lat_b = np.sin(lat_b)
    cos_lat_a = np.cos(lat_a)
    cos_lat_b = np.cos(lat_b)

    num1 = cos_lat_b * sin_difference_long
    num2 = cos_lat_a * sin_lat_b - sin_lat_a * cos_lat_b * cos_difference_long
    denominator = sin_lat_a * sin_lat_b + cos_lat_a * cos_lat_b * cos_difference_long

    return np.arctan2(np.hypot(num1, num2), denominator)/faq

def wrap_mean(array):
    """Works out the mean location taking into account mean of 359 and 1 is 0."""
    if (np.max(array)-np.min(array)>=180) and len(np.where((array>90) & (array<270))[0])==0:
        left=[]
        right=[]
        for k in array:
            if k<180:
                right.append(k)
            else:
                left.append(k)
        left_avg=np.mean(left)-360
        right_avg=np.mean(right)
        avg=np.mean([left_avg,right_avg])
        if avg<0:
            avg+=360
    else:
        avg=np.mean(array)
    return avg

def redshift_projected_unscaled_separation(x, y):
    """Works out the projected distances in Mpc (as apoosed to arcseconds)."""
    ra1, dec1, v1 = x
    ra2, dec2, v2 = y
    separations = calculate_angular_seperation(ra1, dec1, ra2, dec2)
    theta = (np.pi/180) * (separations/2.0)
    v_averages = (v1 + v2)/2.0
    on_sky_distances_mpc = np.sin(theta) * (v_averages)
    return on_sky_distances_mpc
    
def redshift_catalog_mean(X: Union[np.array, pd.DataFrame], 
                          column_names: str, 
                          wrap_columns: str = ['ra']) -> np.array:
    """Works out the variable/field means of a redshift catalog dataset taking into account fields of cyclic types using a wrap_mean approach."""

    if column_names is None:
        if type(X) != pd.DataFrame:
            raise TypeError("X is not a Pandas DataFrame, please either provide X as a Pandas DataFrame object or provide a column_name list of column names")

    if column_names is None:
        if not (set(wrap_columns) <= set(X.columns)):
            raise ValueError("Provided set of column names in X do not match names of wrap/cyclical column names wrap_columns.")
    else:
        if not (set(wrap_columns) <= set(column_names)):
            raise ValueError("Provided set of column names in column_names do not match names of wrap/cyclical column names wrap_columns.")
    
    if type(X) is np.ndarray:
        if len(column_names) != X.shape[1]:
            raise ValueError("Provided list of column names is not of the same length as the number of fields/variables present in dataset X. Please state the full set of column names for the dataset in the order they appear.")
        X = pd.DataFrame(X, columns=column_names)

    list_of_means = []

    for colname in X.columns:
        if colname in wrap_columns:
            list_of_means.append(wrap_mean(X[colname]))
        else:
            list_of_means.append(np.mean(X[colname]))

    return np.array(list_of_means)

def integrate(lower_bound,upper_bound,function):
    """Simple integration method which is faster than the numpy methods."""
    d_x = 1e-5
    x_values = np.arange(lower_bound, upper_bound, d_x)
    y_values = function(x_values)
    ysum = np.sum(y_values)
    val = ysum * d_x
    return val
