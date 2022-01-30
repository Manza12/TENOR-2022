import numpy as np
import pretty_midi as pm
from MIDISynth.music import Piece, Note
from MIDISynth.pianoroll import create_piano_roll
from MIDISynth.plot import plot_piano_roll


# Parameters
time_resolution = 0.05

midi = pm.PrettyMIDI('mozart_2.mid')

piece = Piece('Mozart Sonata 8', final_rest=2.)

for note in midi.instruments[0].notes:
    piece.notes.append(Note(note.pitch, 127, start_seconds=note.start, end_seconds=note.end))

frequency_vector = 440 * 2**((np.arange(21., 108.) - 69.) / 12)
time_vector = np.arange(0., piece.duration(), time_resolution)
piano_roll = create_piano_roll(piece, frequency_vector, time_vector)

fig = plot_piano_roll(piano_roll, frequency_vector, time_vector, show=True, block=False, fig_size=(600, 360))
# fig.subplots_adjust(0.17, 0.24, 0.98, 0.94)
fig.axes[0].set_xlim([1.5/time_resolution, 14.5/time_resolution])
fig.axes[0].set_ylim([20, 60])


if __name__ == '__main__':
    pass

