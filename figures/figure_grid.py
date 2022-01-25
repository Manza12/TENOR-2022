import numpy as np
from plot import plot_timestamps

# Parameters
threshold = 0.05
min_cand = 0.01
max_cand = 1.
res_cand = 0.001
start_cand = 0.5

# Timestamps
timestamps = np.array([0., 0.98, 1.52])

# Plot
plot_timestamps(timestamps, threshold, min_cand, max_cand, res_cand, start_cand, slider=False, box='tight')

if __name__ == '__main__':
    pass
