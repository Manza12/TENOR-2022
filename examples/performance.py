import numpy as np
from acds import compute_acds
from time import time

# Parameters
threshold = 0.05
min_cand = 0.2
max_cand = 1.
res_cand = 0.001
center = True

# N = 20

# Input
# stretch = np.arange(N) + np.random.random(N)*threshold
stretch = np.array((0, 0.98, 1.52))

# Loop
L = 1000
start = time()
for i in range(L):
    acds, acds_errors, acds_multiples, acds_durations = compute_acds(stretch, threshold, min_cand, max_cand, res_cand,
                                                                     center=True, plot=False)
print('Time for ACDs: %.6f' % ((time() - start) / L))

if __name__ == '__main__':
    pass
