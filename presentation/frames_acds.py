from acds import compute_acds
import numpy as np

threshold = 0.05
timestamps = np.array([0, 1.018, 1.531, 2.061, 2.888, 3.179, 4.286])

for i in range(len(timestamps) - 2):
    result = compute_acds(timestamps[i: i+3], threshold, min_cand=0.2, max_cand=1.)
    print('Frame', i+1)
    print('ACDs:', result[0])
    print('errors:', result[1])
