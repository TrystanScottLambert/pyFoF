"""Tests for data validation module."""

import unittest
import warnings
import pandas as pd
import numpy as np

from pyFoF.data_validation import (
    validate_ra,
    validate_dec,
    validate_velocity,
    validate_magnitude,
    validate_astronomical_data,
)


class TestRAValidation(unittest.TestCase):
    """Tests for RA validation."""
    
    def test_valid_ra(self):
        """Test valid RA values."""
        ra = pd.Series([0, 180, 359.9])
        self.assertTrue(validate_ra(ra))
    
    def test_invalid_ra_negative(self):
        """Test negative RA raises warning."""
        ra = pd.Series([-1, 180])
        with self.assertWarns(UserWarning):
            self.assertFalse(validate_ra(ra))
    
    def test_invalid_ra_too_large(self):
        """Test RA > 360 raises warning."""
        ra = pd.Series([360, 400])
        with self.assertWarns(UserWarning):
            self.assertFalse(validate_ra(ra))
    
    def test_ra_with_nans(self):
        """Test RA validation handles NaN values."""
        ra = pd.Series([180, np.nan, 200])
        self.assertTrue(validate_ra(ra))


class TestDecValidation(unittest.TestCase):
    """Tests for Dec validation."""
    
    def test_valid_dec(self):
        """Test valid Dec values."""
        dec = pd.Series([-90, 0, 45, 90])
        self.assertTrue(validate_dec(dec))
    
    def test_invalid_dec_too_low(self):
        """Test Dec < -90 raises warning."""
        dec = pd.Series([-91, 0])
        with self.assertWarns(UserWarning):
            self.assertFalse(validate_dec(dec))
    
    def test_invalid_dec_too_high(self):
        """Test Dec > 90 raises warning."""
        dec = pd.Series([0, 91])
        with self.assertWarns(UserWarning):
            self.assertFalse(validate_dec(dec))


class TestVelocityValidation(unittest.TestCase):
    """Tests for velocity validation."""
    
    def test_valid_velocity(self):
        """Test valid positive velocities."""
        vel = pd.Series([0, 100, 1000])
        self.assertTrue(validate_velocity(vel))
    
    def test_negative_velocity_warning(self):
        """Test negative velocity raises warning."""
        vel = pd.Series([-100, 100])
        with self.assertWarns(UserWarning):
            self.assertFalse(validate_velocity(vel))
    
    def test_negative_velocity_allowed(self):
        """Test negative velocity allowed when specified."""
        vel = pd.Series([-100, 100])
        self.assertTrue(validate_velocity(vel, allow_negative=True))


class TestMagnitudeValidation(unittest.TestCase):
    """Tests for magnitude validation."""
    
    def test_valid_magnitude(self):
        """Test valid magnitude values."""
        mag = pd.Series([0, 10, 20])
        self.assertTrue(validate_magnitude(mag))
    
    def test_bright_magnitude_warning(self):
        """Test very bright magnitude raises warning."""
        mag = pd.Series([-10, 10])
        with self.assertWarns(UserWarning):
            self.assertFalse(validate_magnitude(mag))
    
    def test_faint_magnitude_warning(self):
        """Test very faint magnitude raises warning."""
        mag = pd.Series([10, 35])
        with self.assertWarns(UserWarning):
            self.assertFalse(validate_magnitude(mag))
    
    def test_custom_magnitude_range(self):
        """Test custom magnitude range."""
        mag = pd.Series([1, 2, 3])
        self.assertTrue(validate_magnitude(mag, min_mag=0, max_mag=5))


class TestAstronomicalDataValidation(unittest.TestCase):
    """Tests for combined astronomical data validation."""
    
    def test_valid_data(self):
        """Test validation with valid data."""
        df = pd.DataFrame({
            'ra': [150.0, 200.0],
            'dec': [30.0, -20.0],
            'velocity': [5000, 10000],
            'magnitude': [18.5, 20.0]
        })
        results = validate_astronomical_data(df)
        self.assertTrue(all(results.values()))
    
    def test_column_detection(self):
        """Test automatic column detection."""
        df = pd.DataFrame({
            'RA': [150.0],
            'DEC': [30.0],
        })
        results = validate_astronomical_data(df)
        self.assertIn('ra', results)
        self.assertIn('dec', results)
    
    def test_invalid_ra_in_dataframe(self):
        """Test validation catches invalid RA in DataFrame."""
        df = pd.DataFrame({
            'ra': [400.0],  # Invalid
            'dec': [30.0],
        })
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = validate_astronomical_data(df)
            self.assertFalse(results['ra'])
            self.assertEqual(len(w), 1)


if __name__ == '__main__':
    unittest.main()
