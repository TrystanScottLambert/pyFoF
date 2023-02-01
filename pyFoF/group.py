"""Main group class."""

import numpy as np
import pylab as plt
import astropy.units as u
import astropy.constants as constants
from astropy.coordinates import SkyCoord
from .utils import wrap_mean
from .survey import Survey

class Group:
    """group_properties"""
    def __init__(self, members: np.ndarray, survey: Survey, weights: np.ndarray = None) -> None:
        """initializing"""
        self.members = members
        self.weights = weights
        self.survey = survey
        self.calculate_positional_properties()
        self.velocity_dispersion = np.std(self.survey.data_frame['vel'][self.members])
        self.number_of_members = len(self.members)

    def calculate_positional_properties(self) -> None:
        """returns the position of the group"""
        self.ra = wrap_mean(self.survey.data_frame['ra'][self.members])
        self.dec = np.mean(self.survey.data_frame['dec'][self.members])
        self.vel = np.mean(self.survey.data_frame['vel'][self.members])
        self.redshift = self.vel / constants.c.to(u.km/u.s).value
        self.comoving_distance = self.survey.cosmology.comoving_distance(self.redshift).value
        self.luminosity_distance = self.survey.cosmology.luminosity_distance(self.redshift).value

        coords = SkyCoord(
            ra = self.ra * u.deg, dec = self.dec * u.deg, distance = self.luminosity_distance*u.Mpc)
        self.galactic_l = coords.galactic.l.value
        self.galactic_b = coords.galactic.b.value
        self.equi_x = coords.cartesian.x.value
        self.equi_y = coords.cartesian.y.value
        self.equi_z = coords.cartesian.z.value

    def quicklook(self):
        """Quickly shows what the groups look like and that they are correct."""
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax.scatter(
            self.survey.data_frame['ra'][self.members],
            self.survey.data_frame['dec'][self.members],
            s = 3, c = 'k')
        ax.set_xlabel('RA', fontsize = 13)
        ax.set_ylabel('Dec', fontsize = 13)
        ax1 = fig.add_subplot(212)
        ax1.scatter(
            self.survey.data_frame['vel'][self.members],
            self.survey.data_frame['dec'][self.members],
            s = 3, c = 'k')
        ax1.set_xlabel('Vel', fontsize = 13)
        ax1.set_ylabel('Dec', fontsize = 13)
        plt.show()

    def quicklook_3d(self):
        """Quick look at the group data in 3d."""
        redshifts = self.survey.data_frame['vel'].values / constants.c.to(u.km/u.s).value
        distances = self.survey.cosmology.luminosity_distance(redshifts).value
        fig = plt.figure()
        coords = SkyCoord(
            ra = self.survey.data_frame['ra'][self.members]*u.deg,
            dec = self.survey.data_frame['dec'][self.members]*u.deg,
            distance = distances * u.Mpc)
        ax = fig.add_subplot(projection='3d')
        ax.scatter(
            coords.cartesian.x.value,
            coords.cartesian.y.value,
            coords.cartesian.z.value)
        plt.show()
