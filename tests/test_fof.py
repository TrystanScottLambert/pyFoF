"""Tests for the fof module."""

import unittest
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from pyFoF import fof
from pyFoF import data
from pyFoF import survey


class TestFindingFriends(unittest.TestCase):
    """Running test to check Finding Friends is working" for around a point and around a galaxy."""

    def test_simple(self):
        """Checking if two points will be found as friends."""
        infile = '/home/trystan/Desktop/Work/pyFoF/data/Test_Data/Test_Cat.tbl'
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        my_data = data.read_data(infile)
        test_survey = survey.Survey(my_data, cosmo, 11.75)
        test_survey.convert_z_into_cz('zcmb')
        test_run = fof.Trial(test_survey, d_0=0.56, v_0=350., v_max = 1000., d_max = 2.0)
        friends = test_run.find_friends_from_point(
            3.54946, -33.11417, test_survey.data_frame['vel'][0],
            np.arange(len(test_survey.data_frame)))
        np.testing.assert_array_equal(friends, np.array([0,1,2,3]))
        
        friends = test_run.find_friends_of_galaxy(1, np.arange(len(test_survey.data_frame)))
        np.testing.assert_array_equal(friends, np.array([0,1,2,3]))

    def test_isolated(self):
        """Looking for friends around a galaxy which is completely isolated."""
        infile = '/home/trystan/Desktop/Work/pyFoF/data/Test_Data/Test_Cat.tbl'
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        my_data = data.read_data(infile)
        test_survey = survey.Survey(my_data, cosmo, 11.75)
        test_survey.convert_z_into_cz('zcmb')
        test_run = fof.Trial(test_survey, d_0=0.56, v_0=350., v_max = 1000., d_max = 2.0)
        friends = test_run.find_friends_from_point(
            test_survey.data_frame['ra'][5],
            test_survey.data_frame['dec'][5],
            test_survey.data_frame['vel'][5],
            np.arange(len(test_survey.data_frame)))
        np.testing.assert_array_equal(friends, np.array([5]))

        friends = test_run.find_friends_of_galaxy(5, np.arange(len(test_survey.data_frame)))
        np.testing.assert_array_equal(friends, np.array([5]))
    
    def test_groups_in_correct_position(self):
        """Finding friends and checking that indicies are correct"""
        infile = '/home/trystan/Desktop/Work/pyFoF/data/Test_Data/Test_Cat.tbl'
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        my_data = data.read_data(infile)
        test_survey = survey.Survey(my_data, cosmo, 11.75)
        test_survey.convert_z_into_cz('zcmb')
        test_run = fof.Trial(test_survey, d_0=0.56, v_0=350., v_max = 1000., d_max = 2.0)
        friends = test_run.find_friends_from_point(33.54946, 33.11417, test_survey.data_frame['vel'][6], np.arange(len(test_survey.data_frame)))
        np.testing.assert_array_equal(friends, np.array([6,7,8]))

        friends = test_run.find_friends_of_galaxy(6, np.arange(len(test_survey.data_frame)))
        np.testing.assert_array_equal(friends, np.array([6,7,8]))

if __name__ == '__main__':
    unittest.main()
