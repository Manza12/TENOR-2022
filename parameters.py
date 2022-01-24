import numpy as np

# aGCD
THRESHOLD = 0.05  # in seconds
MIN_CANDIDATE = 0.001  # in seconds
MAX_CANDIDATE = 1.  # in seconds
RESOLUTION = 0.001  # in seconds

# aCD
min_acd = 0.2  # in seconds

candidates = np.arange(MIN_CANDIDATE, MAX_CANDIDATE + RESOLUTION, RESOLUTION)
candidates = np.expand_dims(candidates, 0)

candidates_acd = candidates[candidates > min_acd]
candidates_acd = np.expand_dims(candidates_acd, 0)
candidates_acd_transposed = np.expand_dims(candidates_acd[0], 1)

# Graph parameters
MAXIMUM_TEMPO_VARIATION = 20.  # in percent
MAXIMUM_ERROR = THRESHOLD  # in seconds
WEIGHT_ERROR = 1.
WEIGHT_TEMPO_VARIATION = 1.
