"""Tests for the Utils module."""

import sys
import unittest
import pylab as plt
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
sys.path.append("../")
from pyFoF import utils as pu

class TestAngSep(unittest.TestCase):
    """Test class for calculate_angular_separation."""

    def test_angsep_array(self):
        """Testing if calculate_angular_separation works with arrays."""
        number_of_points = 1000
        right_ascentions_a = (np.random.random(number_of_points) * 2 * np.pi) * (180./np.pi)
        declinations_a = np.arcsin(np.random.random(number_of_points)*2 - 1)
        right_ascentions_b = (np.random.random(number_of_points) * 2 * np.pi) * (180./np.pi)
        declinations_b = np.arcsin(np.random.random(number_of_points)*2 - 1)
        angsep_distances = pu.calculate_angular_seperation(
            right_ascentions_a, declinations_a, right_ascentions_b, declinations_b)

        c_a = SkyCoord(ra = right_ascentions_a * u.deg, dec = declinations_a * u.deg)
        c_b = SkyCoord(ra = right_ascentions_b * u.deg, dec = declinations_b * u.deg)
        c_distances = c_a.separation(c_b)
        skycoord_distances = c_distances.value
        for i in range(number_of_points):
            self.assertAlmostEqual(angsep_distances[i], skycoord_distances[i])


if __name__ == '__main__':
    unittest.main()
