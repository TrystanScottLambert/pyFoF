"""pyFoF: A Python library for Friends-of-Friends clustering."""

from pyFoF.data_handling import read_data, read_in_fits_table, read_in_iaupac_table
from pyFoF.data_validation import (
    validate_astronomical_data,
    validate_ra,
    validate_dec,
    validate_velocity,
    validate_magnitude,
)
from pyFoF.survey import Survey
from pyFoF.fof import FoF

__all__ = [
    "read_data",
    "read_in_fits_table",
    "read_in_iaupac_table",
    "validate_astronomical_data",
    "validate_ra",
    "validate_dec",
    "validate_velocity",
    "validate_magnitude",
    "Survey",
    "FoF",
]
