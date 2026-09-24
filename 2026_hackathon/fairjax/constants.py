"""Constants shared with ``fair.constants``."""

import numpy as np

#: Years for CO2 to double at 1% per year -- the TCR ramp length.
DOUBLING_TIME_1PCT = np.log(2) / np.log(1.01)
