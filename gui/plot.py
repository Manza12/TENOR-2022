import numpy as np
import matplotlib.ticker as tick

from MIDISynth.plot import format_freq, format_time
from MIDISynth.utils import hertz_to_midi


def plot_piano_roll(ax, a, t, f, v_min=0, v_max=127, c_map='Greys', freq_type='int', freq_label='MIDI numbers',
                    time_label='Time (s)'):
    notes_vector = hertz_to_midi(f).astype(np.int)

    ax.imshow(a, cmap=c_map, aspect='auto', vmin=v_min, vmax=v_max, origin='lower', interpolation='none')

    # Freq axis
    ax.yaxis.set_major_formatter(tick.FuncFormatter(lambda x, pos: format_freq(x, pos, notes_vector, freq_type)))

    # Time axis
    ax.xaxis.set_major_formatter(tick.FuncFormatter(lambda x, pos: format_time(x, pos, t)))

    # Labels
    ax.set_xlabel(time_label)
    ax.set_ylabel(freq_label)


def plot_timestamps_radio(timestamps, acds, threshold=0.05, fig_size=(4., 1.2), save=False, radio=True, box=None,
                          notes=()):
    n = len(timestamps)

    if len(notes) == 0:
        notes = np.zeros(n)

    fig = plt.figure(figsize=fig_size)
    plt.tight_layout()

    ax = fig.gca()
    points = plt.scatter(timestamps, notes)

    if radio:
        plt.subplots_adjust(left=0.2)

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

    def update(val):
        x_lim = ax.get_xlim()
        value = float(val)
        new_ticks = np.arange(timestamps[0], timestamps.max() + value, value)
        ax.set_xticks(new_ticks, minor=False)
        ax.set_xlim(x_lim)
        ax.xaxis.grid(True, which='major')
        plt.draw()

    plt.legend([points, line], ['timestamps', 'threshold'])

    if radio:
        ax_radio = plt.axes([0.05, 0.25, 0.13, 0.5])
        grid_radio = RadioButtons(ax_radio, [str(round(acd, 3)) for acd in acds])

        grid_radio.on_clicked(update)

    update(acds[0])

    if save:
        if not box:
            x_min = 0.
            y_min = 0.
            x_max = fig_size[0]
            y_max = fig_size[1]
            fig.savefig('figure_grid.png', bbox_inches=trans.Bbox([[x_min, y_min], [x_max, y_max]]), transparent=True)
            fig.savefig('figure_grid.eps', bbox_inches=trans.Bbox([[x_min, y_min], [x_max, y_max]]), transparent=True)
        else:
            fig.savefig('figure_grid.png', bbox_inches=box, transparent=True)
            fig.savefig('figure_grid.eps', bbox_inches=box, transparent=True)

    plt.show()

    return fig
