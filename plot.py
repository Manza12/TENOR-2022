import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.transforms as trans
from matplotlib.widgets import Slider, RadioButtons


plt.rcParams.update({"font.sans-serif": ["CMU Serif"], "font.size": 12})


def plot_acds(candidates, errors, acds, acds_errors, threshold,
              save=False, save_name='', fig_size=(5., 2.5), latex=False):
    fig = plt.figure(figsize=fig_size)
    line_error = plt.plot(candidates[:, 0], errors)
    line_threshold = plt.axhline(y=threshold, color='k')
    points_acds = plt.scatter(acds, acds_errors, color='r')

    if latex:
        plt.rcParams.update({"text.usetex": True})

    plt.xlabel('Time (s)')
    plt.ylabel('Error (s)')
    plt.legend([points_acds, line_threshold, line_error[0]],
               ['approximate common divisors', r'threshold ($\tau$ = 0.05 s)', r'$\epsilon_{T}$'])

    plt.tight_layout()

    if save:
        fig.savefig(save_name + '.eps', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.show()

    return fig


def plot_timestamps(timestamps, threshold=0.05, min_cand=0.001, max_cand=1., res_cand=0.001, start_cand=0.1,
                    fig_size=(4., 1.2), save=False, slider=True, box=None, notes=()):
    n = len(timestamps)

    if len(notes) == 0:
        notes = np.zeros(n)

    fig = plt.figure(figsize=fig_size)
    plt.tight_layout()

    ax = fig.gca()
    points = plt.scatter(timestamps, notes)

    if slider:
        plt.subplots_adjust(bottom=0.25)

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

    def update(acd, idx):
        x_lim = ax.get_xlim()
        new_ticks = np.concatenate((np.arange(timestamps[idx], timestamps.min() - acd, -acd),
                                    np.arange(timestamps[idx], timestamps.max() + acd, acd)))
        ax.set_xticks(new_ticks, minor=False)
        ax.set_xlim(x_lim)
        ax.xaxis.grid(True, which='major')

        if slider:
            grid_slider.valtext.set_text(str(acd) + ' s')
            note_slider.valtext.set_text(str(round(timestamps[idx], 3)) + ' s')

    plt.legend([points, line], ['timestamps', 'threshold'])

    if slider:
        ax_slider = plt.axes([0.15, 0.05, 0.65, 0.03])
        grid_slider = Slider(
            ax=ax_slider,
            label='aCD',
            valmin=min_cand,
            valmax=max_cand,
            valinit=start_cand,
            valstep=res_cand
        )

        ax_slider_note = plt.axes([0.15, 0.1, 0.65, 0.03])
        note_slider = Slider(
            ax=ax_slider_note,
            label='timestamp',
            valmin=0,
            valmax=n-1,
            valinit=0,
            valstep=1
        )
        grid_slider.on_changed(lambda val: update(val, note_slider.val))
        note_slider.on_changed(lambda val: update(grid_slider.val, val))

    update(start_cand, 0)

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
