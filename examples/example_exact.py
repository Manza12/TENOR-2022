import numpy as np
from acds import compute_acds

# Parameters
threshold = 0.05
min_cand = 0.01
max_cand = 1.
res_cand = 0.001
start_cand = 0.5
center = True
plot = True

timestamps = np.array([0, 1, 1.5, 2, 2.75, 3, 4])

acds, acds_errors, acds_multiples, acds_durations = \
    compute_acds(timestamps, threshold, min_cand, max_cand, res_cand, center, plot,
                 save=False, save_name='', fig_size=(5., 2.5), latex=False)

if __name__ == '__main__':
    pass
