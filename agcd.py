from parameters import *
from scipy.signal import find_peaks


def compute_agcd(timestamps, plot=False):
    integers = np.round(timestamps / candidates)
    approximations = candidates * integers
    errors_matrix = np.abs(timestamps - approximations)
    errors = np.max(errors_matrix, 0)
    peaks_locations, _ = find_peaks(-errors)
    acds = candidates[0, peaks_locations]
    acds_errors = errors[peaks_locations]
    acds_threshold = acds[acds_errors < THRESHOLD]
    acds_threshold_errors = acds_errors[acds_errors < THRESHOLD]
    agcd = np.max(acds_threshold)
    index = np.argmax(acds_threshold_errors)
    error = acds_threshold_errors[index]

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(candidates[0, :], errors)
        plt.scatter(acds, acds_errors, color='g')
        plt.axhline(y=THRESHOLD, color='k')
        plt.scatter(acds_threshold, acds_threshold_errors, color='r')

        plt.title('Error with respect to candidate')
        plt.xlabel('Time (s)')
        plt.ylabel('Maximum error (s)')
        plt.show()

    return agcd, error


def compute_acds(timestamps, plot=False):
    integers = np.round(timestamps / candidates_acd).astype(np.int8)
    approximations = candidates_acd * integers
    errors_matrix = np.abs(timestamps - approximations)
    errors = np.max(errors_matrix, 0)

    peaks_locations, _ = find_peaks(-errors)

    acds = candidates_acd[0, peaks_locations]
    acds_errors = errors[peaks_locations]
    acds_durations = integers[:, peaks_locations]

    under_threshold = acds_errors < THRESHOLD
    indexes_threshold = np.where(under_threshold)

    acds_threshold = acds[indexes_threshold]
    acds_threshold_errors = acds_errors[indexes_threshold]
    acds_threshold_durations = acds_durations[:, indexes_threshold]
    acds_threshold_durations = np.diff(acds_threshold_durations[:, 0, :], axis=0)

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(candidates_acd[0, :], errors)
        plt.scatter(acds, acds_errors, color='g')
        plt.axhline(y=THRESHOLD, color='k')
        plt.scatter(acds_threshold, acds_threshold_errors, color='r')

        plt.title('Error with respect to candidate')
        plt.xlabel('Time (s)')
        plt.ylabel('Maximum error (s)')
        plt.show()

    return acds_threshold, acds_threshold_errors, acds_threshold_durations
