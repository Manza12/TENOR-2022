import numpy as np
from agcd import compute_acds

timestamps = np.expand_dims(np.array([0., 0.4, 0.8, 1.1, 1.21]), 1)
acds_threshold, acds_threshold_errors, acds_threshold_durations = compute_acds(timestamps, plot=True)
