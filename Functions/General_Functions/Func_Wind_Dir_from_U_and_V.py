"""
Function for calculating the wind direction from u and v wind components.
"""

import numpy as np

def wind_dir(u, v):
    """
    Function for calculating the wind direction from u and v wind speed components.
    Generated with the help of ChatUiT.
    0 degrees is north, 90 degrees is east, 180 degrees is south, and 270 degrees is west.
    """

    angle = 270 - np.arctan2(u, v) / np.pi * 180

    return angle % 360  # the modulo leaves the value intact if it is positive and adds 360 if it is negative

# end def