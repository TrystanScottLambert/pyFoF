"""Tests for the Utils module."""

import sys
import unittest
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from scipy.integrate import quad
#sys.path.append("../")
from pyFoF import utils

class TestAngSep(unittest.TestCase):
    """Test class for calculate_angular_separation."""

    def test_angsep_array(self):
        """Testing if calculate_angular_separation works with arrays."""
        number_of_points = 1000
        right_ascentions_a = (np.random.random(number_of_points) * 2 * np.pi) * (180./np.pi)
        declinations_a = np.arcsin(np.random.random(number_of_points)*2 - 1)
        right_ascentions_b = (np.random.random(number_of_points) * 2 * np.pi) * (180./np.pi)
        declinations_b = np.arcsin(np.random.random(number_of_points)*2 - 1)
        angsep_distances = utils.calculate_angular_seperation(
            right_ascentions_a, declinations_a, right_ascentions_b, declinations_b)

        c_a = SkyCoord(ra = right_ascentions_a * u.deg, dec = declinations_a * u.deg)
        c_b = SkyCoord(ra = right_ascentions_b * u.deg, dec = declinations_b * u.deg)
        c_distances = c_a.separation(c_b)
        skycoord_distances = c_distances.value
        for i in range(number_of_points):
            self.assertAlmostEqual(angsep_distances[i], skycoord_distances[i])

    def test_angsep_single(self):
        """Testing that single points should give a distance too."""
        number_of_points = 100
        right_ascentions_a = (np.random.random(number_of_points) * 2 * np.pi) * (180./np.pi)
        declinations_a = np.arcsin(np.random.random(number_of_points)*2 - 1)
        right_ascentions_b = (np.random.random(number_of_points) * 2 * np.pi) * (180./np.pi)
        declinations_b = np.arcsin(np.random.random(number_of_points)*2 - 1)

        for i in range(number_of_points):
            c_a = SkyCoord(ra = right_ascentions_a[i] * u.deg, dec = declinations_a[i] * u.deg)
            c_b = SkyCoord(ra = right_ascentions_b[i] * u.deg, dec = declinations_b[i] * u.deg)
            sep = c_a.separation(c_b)
            self.assertAlmostEqual(sep.value, utils.calculate_angular_seperation(
                right_ascentions_a[i], declinations_a[i], right_ascentions_b[i], declinations_b[i]))

class TestWrapMean(unittest.TestCase):
    """Test class for wrap_mean."""

    def test_standard(self):
        """Testing if functions work."""
        array = np.array([359.2, 359, 358.3, 1, 2, 1.2])
        self.assertAlmostEqual(utils.wrap_mean(array), 0.11666666666666665)

class TestIntegrate(unittest.TestCase):
    """Test class for integrate function."""

    def test_integrate(self):
        """testing if integrate works."""
        def func(x):
            """Simple test function."""
            return x*2 + x + 7
        functions = [np.sin, np.cos, func]
        for _function in functions:
            scipy_integral, _ = quad(_function, -np.pi, np.pi/4)
            self.assertAlmostEqual(scipy_integral, utils.integrate(-np.pi, np.pi/4, _function),
                                    places = 4)

if __name__ == '__main__':
    unittest.main()
