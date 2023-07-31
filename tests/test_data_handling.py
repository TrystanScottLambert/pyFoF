import unittest
from astropy.table import Table
from pandas._testing import assert_frame_equal
import pandas as pd
import numpy as np
from pyFoF import data_handling

class TestDataHandlingFunctions(unittest.TestCase):

    def setUp(self):
        """Set up the data for the test cases"""
        self.example_fits_file = './data/Test_Data/Test_Data_Handling.fits' 
        self.example_iaupac_file = './data/Test_Data/Test_Data_Handling.tbl'
        self.data_frame = pd.DataFrame({'idname': [94967, 94976, 94982], 
                                     'ra': [3.62163, 3.54946, 3.53893], 
                                     'dec': [-33.11417, -32.39796, -32.71177],
                                     'zcmb': [0.03029, 0.05788, 0.06229],
                                     'zhelio': [0.0312, 0.0588, 0.0632],
                                     'W1': [15.275, 14.542, 16.445],
                                     'eW1': [0.056, 0.034, 0.060]})

    def test_read_in_fits_table(self):
        """Test that reading in a FITS table produces the expected DataFrame."""

        df_from_fits = data_handling.read_in_fits_table(self.example_fits_file)
        assert_frame_equal(self.data_frame, df_from_fits, check_dtype=False)

    def test_read_in_iaupac_table(self):
        """Test that reading in an IAUPAC table produces the expected DataFrame."""

        test_dataframe = self.data_frame.astype('string')
        df_from_iaupac = data_handling.read_in_iaupac_table(self.example_iaupac_file)
        assert_frame_equal(test_dataframe, df_from_iaupac, check_dtype=False)

    def test_check_file_type(self):
        """Test that file type detection works correctly for FITS and IAUPAC files."""

        fits_ext = data_handling.check_file_type(self.example_fits_file)
        self.assertEqual(fits_ext, "fits")

        iaupac_ext = data_handling.check_file_type(self.example_iaupac_file)
        self.assertEqual(iaupac_ext, "tbl")

    def test_infer_dtype(self):
        """Test that the infer_dtype function correctly identifies the types of string inputs."""

        self.assertEqual(data_handling.infer_dtype("1"), int)
        self.assertEqual(data_handling.infer_dtype("1.0"), float)
        self.assertIn(data_handling.infer_dtype("string"), [str, object])


    def test_auto_convert_df_types(self):
        """Test that the auto_convert_df_types function correctly converts column types in a DataFrame."""

        data = {'a': ['1', '2', '3'],
                'b': ['1.1', '2.2', '3.3'],
                'c': ['string1', 'string2', 'string3']}
        df = pd.DataFrame(data)

        df_converted = data_handling.auto_convert_df_types(df)

        self.assertEqual(df_converted.dtypes['a'], int)
        self.assertEqual(df_converted.dtypes['b'], float)
        self.assertIn(df_converted.dtypes['c'], [str, object]) 

    def test_read_data(self):
        """Test that reading in data works correctly for both FITS and IAUPAC files, producing the expected DataFrame."""

        df_fits = data_handling.read_data(self.example_fits_file)
        assert_frame_equal(self.data_frame, df_fits)

        df_iaupac = data_handling.read_data(self.example_iaupac_file)
        assert_frame_equal(self.data_frame, df_iaupac)

if __name__ == '__main__':
    unittest.main()
