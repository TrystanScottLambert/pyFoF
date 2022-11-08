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
