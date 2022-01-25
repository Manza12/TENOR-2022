from parameters import *
from scipy.signal import find_peaks
import matplotlib as mpl
mpl.rc('font', family='CMU Serif')


def plot_timestamps(timestamps, notes, threshold):
    timestamps = timestamps[0]
    assert len(timestamps) == len(notes)
    n = len(timestamps)

    import matplotlib.pyplot as plt
    import matplotlib.lines as lines
    from matplotlib.widgets import Slider

    fig = plt.figure()
    ax = fig.gca()
    plt.scatter(timestamps, notes)
    plt.subplots_adjust(bottom=0.25)

    for i in range(n):
        line = lines.Line2D([timestamps[i] - threshold, timestamps[i] + threshold], [notes[i], notes[i]], color='r')
        ax.add_line(line)

    ax_slider = plt.axes([0.15, 0.1, 0.65, 0.03])
    grid_slider = Slider(
        ax=ax_slider,
        label='aCD',
        valmin=min_acd,
        valmax=MAX_CANDIDATE,
        valinit=min_acd,
    )

    def update(val):
        ax.set_xticks(np.arange(0., timestamps.max(), val), minor=False)
        ax.xaxis.grid(True, which='major')

    grid_slider.on_changed(update)

    plt.show()

    return fig


def plot_acds(acds, acds_errors, errors, acds_threshold, acds_threshold_errors, latex=True, save=False):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(5., 2.5))

    line_error = plt.plot(candidates_acd[0, :], errors)
    if not acds.all():
        plt.scatter(acds, acds_errors, color='g')
    line_threshold = plt.axhline(y=THRESHOLD, color='k')
    points_acds = plt.scatter(acds_threshold, acds_threshold_errors, color='r')

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
        plot_acds(acds, acds_errors, errors, acds_threshold, acds_threshold_errors)

    return agcd, error


def compute_acds(timestamps, plot=False, center=True, latex=True, save=False):
    if center:
        timestamps = timestamps - timestamps[0]
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
        plot_acds(acds, acds_errors, errors, acds_threshold, acds_threshold_errors, latex=latex, save=save)

    return acds_threshold, acds_threshold_errors, acds_threshold_durations


def compute_acds_transposed(timestamps):
    integers = np.round(timestamps / candidates_acd_transposed).astype(np.int8)
    approximations = candidates_acd_transposed * integers
    errors_matrix = np.abs(timestamps - approximations)
    errors = np.max(errors_matrix, 1)

    peaks_locations, _ = find_peaks(-errors)

    acds = candidates_acd[0, peaks_locations]
    acds_errors = errors[peaks_locations]
    acds_durations = integers[peaks_locations, :]

    under_threshold = acds_errors < THRESHOLD
    indexes_threshold = np.where(under_threshold)

    acds_threshold = acds[indexes_threshold]
    acds_threshold_errors = acds_errors[indexes_threshold]
    acds_threshold_durations = acds_durations[indexes_threshold, :]
    acds_threshold_durations = np.diff(acds_threshold_durations[0, :, :], axis=1)

    return acds_threshold, acds_threshold_errors, acds_threshold_durations


def compute_acds_multiples(stretch, notes=(), plot=False):
    centered_stretch = stretch - stretch[0, 0]
    integers = np.round(centered_stretch / candidates_acd_transposed).astype(np.int8)
    approximations = candidates_acd_transposed * integers
    errors_matrix = np.abs(centered_stretch - approximations)
    errors = np.max(errors_matrix, 1)

    peaks_locations, _ = find_peaks(-errors)

    acds = candidates_acd[0, peaks_locations]
    acds_errors = errors[peaks_locations]
    acds_durations = integers[peaks_locations, :]

    under_threshold = acds_errors < THRESHOLD
    indexes_threshold = np.where(under_threshold)

    acds_threshold = acds[indexes_threshold]
    acds_threshold_errors = acds_errors[indexes_threshold]
    acds_threshold_durations = acds_durations[indexes_threshold, :]

    if plot:
        plot_timestamps(centered_stretch, notes, THRESHOLD)
        plot_acds(acds, acds_errors, errors, acds_threshold, acds_threshold_errors)

    return np.flip(acds_threshold), np.flip(acds_threshold_errors), np.flip(acds_threshold_durations[0], axis=0)
