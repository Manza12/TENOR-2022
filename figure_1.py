import numpy as np
from agcd import compute_acds

timestamps = np.expand_dims(np.array([0., 0.98, 1.52]), 1)
acds_threshold, acds_threshold_errors, acds_threshold_durations = compute_acds(timestamps, plot=True, save=True)
