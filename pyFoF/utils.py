"""Utilities which are needed for the package."""

import numpy as np

def calculate_angular_seperation(longitude_a, latitude_a, longitude_b, latitude_b):
    """
    Determines the angular separation between two points, or two catalogs.
    The inputs must be in degrees and can be floats or numpy arrays of floats.
    """

    faq = (np.pi/180)
    longitude_a, longitude_b, latitude_a, latitude_b = longitude_a*faq, longitude_b*faq, latitude_a*faq, latitude_b*faq
    sdlon = np.sin(longitude_b - longitude_a)
    cdlon = np.cos(longitude_b - longitude_a)
    sin_latitude_a = np.sin(latitude_a)
    sin_latitude_b = np.sin(latitude_b)
    cos_latitude_a = np.cos(latitude_a)
    cos_latitude_b = np.cos(latitude_b)

    num1 = cos_latitude_b * sdlon
    num2 = cos_latitude_a * sin_latitude_b - sin_latitude_a * cos_latitude_b * cdlon
    denominator = sin_latitude_a * sin_latitude_b + cos_latitude_a * cos_latitude_b * cdlon

    return np.arctan2(np.hypot(num1, num2), denominator)/faq
