"""Main group class."""

from typing import Tuple
import numpy as np
import astropy.units as u
import astropy.constants as constants
from astropy.coordinates import SkyCoord
from utils import wrap_mean
from survey import Survey

class Group:
    """group_properties"""
    def __init__(self, members: np.ndarray, survey: Survey, weights: np.ndarray = None) -> None:
        """initializing"""
        self.members = members
        self.number_of_members = len(self.members)
        self.weights = weights
        self.survey = survey


    def calculate_positional_data(self) -> Tuple[float, float, float]:
        """returns the position of the group"""
        ra = wrap_mean(self.survey.data_frame['ra'][self.members])
        dec = np.mean(self.survey.data_frame['dec'][self.members])
        vel = np.mean(self.survey.data_frame['vel'][self.members])

        coords = SkyCoord(ra = ra * u.deg, dec = dec * u.deg)
        galactic_l = coords.galactic.l.value
        galactic_b = coords.galactic.b.value

        redshift = vel / constants.c.to(u.km/u.s).value
        luminosity_distance = self.survey.cosmology.luminosity_distance(redshift).value
        comoving_distance = self.survey.cosmology.comoving_distance(redshift).value

        return ra, dec, galactic_l, galactic_b, vel, redshift, luminosity_distance, comoving_distance
