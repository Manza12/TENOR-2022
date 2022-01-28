import numpy as np
import pretty_midi as pm
from MIDISynth.music import Piece, Note
from MIDISynth.pianoroll import create_piano_roll
from MIDISynth.plot import plot_piano_roll

midi = pm.PrettyMIDI('mozart_1.mid')

piece = Piece('Mozart Sonata 8', final_rest=2.)

for note in midi.instruments[0].notes:
    piece.notes.append(Note(note.pitch, 127, start_seconds=note.start, end_seconds=note.end))

frequency_vector = 440 * 2**((np.arange(21., 108.) - 69.) / 12)
time_vector = np.arange(0., piece.duration(), 0.001)
piano_roll = create_piano_roll(piece, frequency_vector, time_vector)

plot_piano_roll(piano_roll, frequency_vector, time_vector, show=True, block=True)

if __name__ == '__main__':
    pass
