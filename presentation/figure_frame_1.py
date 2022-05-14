import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines

# Timestamps
_timestamps = np.array([0, 1.018, 1.531])

# Plot
_n = len(_timestamps)
_notes = np.zeros(_n)

_fig = plt.figure(figsize=(3., 1.))

_ax = _fig.gca()
_points = plt.scatter(_timestamps, _notes, marker='|',  s=100)

_ax.set_xlabel('Time (s)')
_ax.get_yaxis().set_ticks([])
if not len(_notes) == 0:
    _ax.set_ylim([np.min(_notes) - 2, np.max(_notes) + 2])
else:
    _ax.set_ylim([-0.1, 0.5])

plt.tight_layout()

_fig.savefig('frame_1.svg', transparent=True)


def plot_timestamps(timestamps, acd, threshold=0.05, fig_size=(4., 1.2), notes=()):
    n = len(timestamps)

    if len(notes) == 0:
        notes = np.zeros(n)

    fig = plt.figure(figsize=fig_size)
    plt.tight_layout()

    ax = fig.gca()
    points = plt.scatter(timestamps, notes)

    ax.set_xlabel('Time (s)')
    ax.get_yaxis().set_ticks([])
    if not len(notes) == 0:
        ax.set_ylim([np.min(notes) - 2, np.max(notes) + 2])
    else:
        ax.set_ylim([-0.1, 0.5])

    line = ()
    for i in range(n):
        line = lines.Line2D([timestamps[i] - threshold, timestamps[i] + threshold], [notes[i], notes[i]], color='r')
        ax.add_line(line)

    x_lim = ax.get_xlim()
    new_ticks = np.concatenate((np.arange(timestamps[0], timestamps.min() - acd, -acd),
                                np.arange(timestamps[0], timestamps.max() + acd, acd)))
    ax.set_xticks(new_ticks, minor=False)
    ax.set_xlim(x_lim)
    ax.xaxis.grid(True, which='major')

    return fig


if __name__ == '__main__':
    a_0 = 0.51
    plot_timestamps(_timestamps, a_0)
    plt.tight_layout()
    plt.savefig('grid_0.svg', transparent=True)

    a_1 = 0.255
    plot_timestamps(_timestamps, a_1)
    plt.tight_layout()
    plt.savefig('grid_1.svg', transparent=True)

    a_2 = 0.212
    plot_timestamps(_timestamps, a_2)
    plt.tight_layout()
    plt.savefig('grid_2.svg', transparent=True)

    plt.show()
