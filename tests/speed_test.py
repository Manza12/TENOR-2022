from acds import compute_acds
import numpy as np
from time import time

N = 1000
timestamps = np.arange(0, 0.5 * N, 0.5) + np.random.rand(N) / 0.025

M = 1000
min_cand = 0.1
max_cand = 1.1
res_cand = 1 / M


if __name__ == '__main__':
    start = time()
    compute_acds(timestamps, min_cand=min_cand, max_cand=max_cand, res_cand=res_cand)
    time_elapsed = time() - start
    print('Time: %.3f ms' % (time_elapsed * 1e3))
    print('Ratio time/MxN (in nano seconds per element): %.3f' % (time_elapsed * 1e9 / (M * N)))
