import numpy as np
import pretty_midi as pm
from plot import plot_timestamps, plot_timestamps_bis

midi = pm.PrettyMIDI('mozart_1.mid')

timestamps = np.zeros(0)
notes = np.zeros(0)

for note in midi.instruments[0].notes:
    timestamps = np.concatenate((timestamps, np.array([note.start])))
    notes = np.concatenate((notes, np.array([note.pitch])))

indexes = np.argsort(timestamps)
timestamps = timestamps[indexes]
notes = notes[indexes]

plot_timestamps_bis(timestamps, notes=notes)

if __name__ == '__main__':
    pass
