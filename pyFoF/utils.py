"""Utilities which are needed for the package."""

import numpy as np

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

def integrate(lower_bound,upper_bound,function):
    """Simple integration method which is faster than the numpy methods."""
    d_x = 1e-7
    x_values = np.arange(lower_bound, upper_bound, d_x)
    y_values = function(x_values)
    ysum = np.sum(y_values)
    val = ysum * d_x
    return val
