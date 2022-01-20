from parameters import *
from agcd import compute_agcd


durations = [1.00, 0.60, 0.30, 0.28, 0.26, 0.24, 0.22, 0.60, 1.00]
timestamps = np.cumsum([0.] + durations)
timestamps = np.expand_dims(timestamps, 1)

stretch = timestamps[0:3, :]

agcd, agcd_error = compute_agcd(stretch)
