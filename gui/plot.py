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
