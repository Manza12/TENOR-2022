import numpy as np
from acds import compute_acds

# Parameters
threshold = 0.05
min_cand = 0.2
max_cand = 1.
res_cand = 0.001
start_cand = 0.5
center = True
plot = True

# [0., 1.018, 1.531, 2.061, 2.888, 3.179, 4.286]
timestamps = np.array([0., 1.018, 1.531])

acds, acds_errors, acds_multiples, acds_durations = \
    compute_acds(timestamps, threshold, min_cand, max_cand, res_cand, center, plot,
                 save=False, save_name='', fig_size=(5., 2.5), latex=False)

for k in range(len(acds)):
    print(k)
    print('aCD:', round(acds[k], 3), 's')
    print('error:', round(acds_errors[k], 3), 's')
    print('multiples:', acds_multiples[k, :])
    print('durations:', acds_durations[k, :])

if __name__ == '__main__':
    pass
