"""Main group class."""

import numpy as np 
import pandas as pd
from typing import Tuple
from utils import wrap_mean

class Group:
    """group_properties"""
    def __init__(self, members: np.ndarray) -> None:
        """initializing"""
        self.members = members
        self.number_of_members = len(self.members)
    
    def position(self, data_frame: pd.DataFrame) -> Tuple[float, float, float]:
        """returns the position of the group"""
        group_ra = wrap_mean(data_frame['ra'][self.members])
        group_dec = np.mean(data_frame['dec'][self.members])
        group_vel = np.mean(data_frame['vel'][self.members])
        return group_ra, group_dec, group_vel
