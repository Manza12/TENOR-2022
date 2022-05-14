import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.signal import find_peaks

mpl.rc('font', family='CMU Serif', size=12)

# Input
_threshold = 0.125
_min_cand = 0.001
_max_cand = 2.5
_res_cand = 0.001

_timestamps = [0., 4.02, 6.05, 7.96]


# Functions
def plot_error_function(candidates, errors,
                        fig_size=(5., 2.5), latex=False):
    fig = plt.figure(figsize=fig_size)
    line_error = plt.plot(candidates[:, 0], errors)

    if latex:
        plt.rcParams.update({"text.usetex": True})

    plt.xlabel('Time (s)')
    plt.ylabel('Error (s)')
    plt.legend([line_error[0]], [r'$\epsilon_{T}$'])

    return fig


def plot_threshold(candidates, errors, threshold,
                   fig_size=(5., 2.5), latex=False):
    fig = plt.figure(figsize=fig_size)
    line_error = plt.plot(candidates[:, 0], errors)
    line_threshold = plt.axhline(y=threshold-0.002, color='k', linewidth=2)

    if latex:
        plt.rcParams.update({"text.usetex": True})

    plt.xlabel('Time (s)')
    plt.ylabel('Error (s)')
    plt.legend([line_threshold, line_error[0]],
               [r'threshold ($\tau$ = %.3f s)' % threshold,
                r'$\epsilon_{T}$'])

    return fig


def plot_candidates(candidates, errors, threshold, x_under_threshold, y_under_threshold,
                    fig_size=(5., 2.5), latex=False):
    fig = plt.figure(figsize=fig_size)
    line_error = plt.plot(candidates[:, 0], errors)
    line_under_threshold = plt.plot(x_under_threshold[:, 0], y_under_threshold, color='r')
    line_threshold = plt.axhline(y=threshold-0.002, color='k', linewidth=2)

    if latex:
        plt.rcParams.update({"text.usetex": True})

    plt.xlabel('Time (s)')
    plt.ylabel('Error (s)')
    plt.legend([line_under_threshold[0], line_threshold, line_error[0]],
               [r'$\{a > 0: \epsilon_{T}(a) \leq \tau\}$',
                r'threshold ($\tau$ = %.3f s)' % threshold,
                r'$\epsilon_{T}$'])

    return fig


def plot_acds_full(candidates, errors, acds, acds_errors, threshold,
                   fig_size=(5., 2.5), latex=False):
    fig = plt.figure(figsize=fig_size)
    line_error = plt.plot(candidates[:, 0], errors)
    line_threshold = plt.axhline(y=threshold, color='k')
    points_acds = plt.scatter(acds, acds_errors, color='r')

    if latex:
        plt.rcParams.update({"text.usetex": True})

    plt.xlabel('Time (s)')
    plt.ylabel('Error (s)')
    plt.legend([points_acds, line_threshold, line_error[0]],
               ['approximate common divisors', r'threshold ($\tau$ = %.3f s)' % threshold, r'$\epsilon_{T}$'])

    return fig


def compute_acds(timestamps, threshold=0.05, min_cand=0.1, max_cand=1., res_cand=0.001,
                 center=True, plot='full', **kwargs):
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

    if plot == 'full':
        plot_acds_full(candidates, errors, acds, acds_errors, threshold, **kwargs)
    elif plot == 'error':
        plot_error_function(candidates, errors, **kwargs)
    elif plot == 'threshold':
        plot_threshold(candidates, errors, threshold, **kwargs)
    elif plot == 'candidates':
        y_under_threshold = errors[errors < threshold]
        x_under_threshold = candidates[errors < threshold]
        plot_candidates(candidates, errors, threshold, x_under_threshold, y_under_threshold, **kwargs)

    return np.flip(acds), np.flip(acds_errors), np.flip(acds_multiples, axis=0), np.flip(acds_durations, axis=0)


# Plots
# Error
compute_acds(_timestamps, _threshold, _min_cand, _max_cand, _res_cand, False, 'error',
             latex=True, fig_size=(6., 3.))
plt.tight_layout()
plt.savefig('error_function.svg', transparent=True)

# Threshold
compute_acds(_timestamps, _threshold, _min_cand, _max_cand, _res_cand, False, 'threshold',
             latex=True, fig_size=(6., 3.))
plt.tight_layout()
plt.savefig('error_threshold.svg', transparent=True)

# Candidates
compute_acds(_timestamps, _threshold, _min_cand, _max_cand, _res_cand, False, 'candidates',
             latex=True, fig_size=(6., 3.))
plt.tight_layout()
plt.savefig('error_function_candidates.svg', transparent=True)

# Full
compute_acds(_timestamps, _threshold, _min_cand, _max_cand, _res_cand, False, 'full',
             latex=True, fig_size=(6., 3.))
plt.tight_layout()
plt.savefig('error_function_full.svg', transparent=True)

plt.show()
