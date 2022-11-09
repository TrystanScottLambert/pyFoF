"""Survey class to keep all survey properties together."""

class Survey:
    """Class representing an individual survey."""
    def __init__(self, data, apparent_mag_limit, fiducial_velocity = 1000):
        self.apparent_magnitude_limit = apparent_mag_limit
        self.fiducial_velocity = fiducial_velocity
