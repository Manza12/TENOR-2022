import numpy as np
import pretty_midi as pm
from plot import plot_timestamps

midi = pm.PrettyMIDI('mozart_2.mid')

timestamps = np.zeros(0)
notes = np.zeros(0)

for note in midi.instruments[0].notes:
    timestamps = np.concatenate((timestamps, np.array([note.start])))
    notes = np.concatenate((notes, np.array([note.pitch])))

# print(timestamps)

plot_timestamps(timestamps - np.min(timestamps), notes=notes)

if __name__ == '__main__':
    pass