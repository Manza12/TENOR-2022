import numpy as np
from scipy.signal import find_peaks


def plot_acds(candidates_acd, errors, acds, acds_errors, threshold, latex=True, save=False):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(5., 2.5))

    line_error = plt.plot(candidates_acd[0, :], errors)
    line_threshold = plt.axhline(y=threshold, color='k')
    points_acds = plt.scatter(acds, acds_errors, color='r')

    if latex:
        plt.rcParams.update({
            "text.usetex": True,
            # "font.sans-serif": ["CMU Serif"]
            })
    # plt.title(r'$\epsilon_{T}(a)$', usetex=True)
    plt.xlabel('Time (s)')
    plt.ylabel('Error (s)')
    # r'$\epsilon_{T}(a)$'
    if latex:
        plt.legend([points_acds, line_threshold, line_error[0]],
                   ['approximate common divisors', r'threshold ($\tau$ = 0.05 s)', r'$\epsilon_{T}$'])

    plt.tight_layout()

    if save:
        fig.savefig('Figure_1.eps', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.show()

    return fig


def compute_acds(timestamps, min_cand=0.001, max_cand=1., resolution_cand=0.001, threshold=0.05,
                 plot=False, center=True, latex=True, save=False):
    candidates_acd = np.expand_dims(np.arange(min_cand, max_cand + resolution_cand, resolution_cand), 0)
    timestamps = np.expand_dims(timestamps, 1)
    if center:
        timestamps = timestamps - timestamps[0]
    integers = np.round(timestamps / candidates_acd).astype(np.int8)
    approximations = candidates_acd * integers
    errors_matrix = np.abs(timestamps - approximations)
    errors = np.max(errors_matrix, 0)

    peaks_locations, _ = find_peaks(-errors)

    acds_potential = candidates_acd[0, peaks_locations]
    acds_potential_errors = errors[peaks_locations]
    acds_potential_durations = integers[:, peaks_locations]

    under_threshold = acds_potential_errors < threshold
    indexes_threshold = np.where(under_threshold)

    acds = acds_potential[indexes_threshold]
    acds_errors = acds_potential_errors[indexes_threshold]
    acds_multiples = acds_potential_durations[:, indexes_threshold][:, 0, :]
    acds_durations = np.diff(acds_multiples, axis=0)

    if plot:
        plot_acds(candidates_acd, errors, acds, acds_errors, threshold, latex, save)

    return acds, acds_errors, acds_multiples, acds_durations


if __name__ == '__main__':
    durations = np.array([399, 710, 105, 891, 402, 102, 397])
    _acds, _acds_errors, _acds_multiples, _acds_durations = \
        compute_acds(durations, min_cand=10, max_cand=1000, resolution_cand=1, threshold=20,
                     plot=True, center=False, latex=False, save=False)
