"""Survey class to keep all survey properties together."""

import pandas as pd
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy import constants as c
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.integrate import quad


class Survey:
    """Class representing an individual survey."""
    def __init__(self, data: pd.DataFrame, cosmology: FlatLambdaCDM,
                 apparent_mag_limit: float,
                 fiducial_velocity:float = 1000.,
                 alpha = -1.02,
                 m_star = -24.2,
                 phi_star = 0.0108):

        """Initializing"""
        self.apparent_magnitude_limit = apparent_mag_limit
        self.fiducial_velocity = fiducial_velocity
        self.data_frame = data
        self.cosmology = cosmology
        self.alpha = alpha
        self.m_star = m_star
        self.phi_star = phi_star
        self.integral = quad(self.shecter_function, -100, self.absolute_mag_lim) # sun's apparent mag = -26.74 so if you're detecting something brighter, you fucked up.
        self._add_idx_information_to_df()

    @property
    def absolute_mag_lim(self) -> float:
        """Absolute magnitude limit."""
        abs_mag_lim = self.apparent_magnitude_limit-25-5*np.log10(
            self.fiducial_velocity / self.cosmology.H0.value)
        return abs_mag_lim

    def shecter_function(self, magnitdues) -> np.ndarray:
        """Shecter Luminosity Function as proposed by Kochanek et al. (2001)."""
        constant = 0.4 * np.log(10) * self.phi_star
        term_1 = 10**(0.4 * (self.alpha+1) * (self.m_star-magnitdues))
        term_2 = np.exp(-10**(0.4 * (self.m_star-magnitdues)))
        return constant * term_1 * term_2

    def m_12(self, v_avg) -> float:
        """Works out average magnitude."""
        return np.array(
            self.apparent_magnitude_limit - 25 - 5*np.log10(v_avg/self.cosmology.H0.value))

    def convert_z_into_cz(self, z_column_name) -> None:
        """Takes the z column and makes a redhsift column in km/s."""
        self.data_frame['vel'] = self.data_frame[z_column_name] * c.c.to(u.km/u.s).value

    def make_mag_column(self, mag_column_name) -> None:
        """Copies the magnitude column and changes it into the needed keyword 'mag' """
        self.data_frame['mag'] = self.data_frame[mag_column_name]

    def _add_idx_information_to_df(self) -> None:
        """Adds necessary id information which will be used by other classes."""
        self.data_frame['fof_ids'] = np.arange(len(self.data_frame))

    def add_positional_information_to_df(self, vel_column_name: str) -> None:
        """Adding positional information including l, b, lum_distance and xyz coords."""
        redshifts = np.array(self.data_frame[vel_column_name]/c.c.to(u.km/u.s).value)
        luminosity_distances = self.cosmology.luminosity_distance(redshifts)
        coords = SkyCoord(
            ra = self.data_frame['ra'] * u.deg,
            dec = self.data_frame['dec'] * u.deg,
            distance = luminosity_distances * u.Mpc
            )
        self.data_frame['luminosity_distance'] = luminosity_distances.value
        self.data_frame['galactic_l'] = coords.galactic.l.value
        self.data_frame['galactic_b'] = coords.galactic.b.value
        self.data_frame['equi_x'] = coords.cartesian.x.value
        self.data_frame['equi_y'] = coords.cartesian.y.value
        self.data_frame['equi_z'] = coords.cartesian.z.value
