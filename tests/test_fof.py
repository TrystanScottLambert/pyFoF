"""Tests for the fof module."""

import sys
import unittest
import numpy as np
from astropy.cosmology import FlatLambdaCDM
sys.path.append("../")
from pyFoF import fof
from pyFoF.data import read_data
#from pyFoF.survey import Survey

class TestFindingFriendsFromPoint(unittest.TestCase):
    """Running test to check Finding Friends is working"""

    def test_simple(self):
        """Checking if two points will be found as friends."""
        infile = '/home/trystan/Desktop/Work/pyFoF/data/Test_Data/Test_Cat.tbl'
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        data = read_data(infile)
        kids = Survey(data, cosmo, 11.75)
        print(kids.data_frame['zcmb'] * 3e5)
        #print(kids.data_frame['zcmb'])
        #kids.convert_z_into_cz('zcmb')
        #test_run = fof.Trial(kids, d_0=0.56, v_0=350., v_max = 1000., d_max = 2.0)
        #friends = test_run.find_friends_from_point(3.54946, -33.11417, kids.data_frame['vel'][0].value)
        np.testing.assert_array_equal(friends, np.array([0,1,2,3]))
    
if __name__ == '__main__':
    unittest.main()
