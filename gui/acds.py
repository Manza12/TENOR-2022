import numpy as np
from scipy.signal import find_peaks


def compute_acds(timestamps, threshold=0.05, min_cand=0.1, max_cand=1., res_cand=0.001, center=True):
    if center:
        timestamps = timestamps - timestamps[0]
    timestamps = np.expand_dims(timestamps, 0)
    candidates = np.expand_dims(np.arange(min_cand, max_cand, res_cand), 1)

    integers = np.round(timestamps / candidates).astype(np.int)
    approximations = candidates * integers
    errors_matrix = np.abs(timestamps - approximations)
    errors = np.max(errors_matrix, 1)

    peaks_locations, _ = find_peaks(-errors)

    minimums = candidates[peaks_locations, 0]
    minimums_errors = errors[peaks_locations]
    minimums_multiples = integers[peaks_locations, :]

    under_threshold = minimums_errors < threshold
    indexes_threshold = np.where(under_threshold)

    acds = minimums[indexes_threshold]
    acds_errors = minimums_errors[indexes_threshold]
    acds_multiples = minimums_multiples[indexes_threshold, :][0, :, :]
    acds_durations = np.diff(acds_multiples, axis=1)

    return np.flip(acds), np.flip(acds_errors), np.flip(acds_multiples, axis=0), np.flip(acds_durations, axis=0)
