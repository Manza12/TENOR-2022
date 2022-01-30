import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pretty_midi as pm
from MIDISynth.music import Piece, Note
from MIDISynth.pianoroll import create_piano_roll
from MIDISynth.plot import plot_piano_roll
from plot import plot_timestamps
from graph import create_polyphonic_graph

# Parameters
frame_size = 2.
hop_size = frame_size / 2

threshold = 0.05
min_cand = 0.15
max_cand = 1.
res_cand = 0.001
center = True

time_resolution = 0.01

error_weight = 0.
tempo_var_weight = 1.

plot_steps = False

# MIDI
midi = pm.PrettyMIDI('chopin_1.mid')

timestamps = np.zeros(0)
notes = np.zeros(0)

for note in midi.instruments[0].notes:
    timestamps = np.concatenate((timestamps, np.array([note.start])))
    notes = np.concatenate((notes, np.array([note.pitch])))

indexes = np.argsort(timestamps)
timestamps = timestamps[indexes]
notes = notes[indexes]

# Graph
graph = create_polyphonic_graph(timestamps, notes=notes, frame_size=frame_size, hop_size=hop_size,
                                threshold=threshold, min_cand=min_cand, max_cand=max_cand, res_cand=res_cand,
                                error_weight=error_weight, tempo_var_weight=tempo_var_weight,
                                plot_steps=plot_steps, start_node=False, final_node=False)

# Frames
frames = graph.graph['frames']

# Grid
selected_frame = 4
frame = frames[selected_frame]

timestamps = np.zeros(0)
notes = np.zeros(0)

for note in midi.instruments[0].notes:
    if note.start in frame:
        timestamps = np.concatenate((timestamps, np.array([note.start])))
        notes = np.concatenate((notes, np.array([note.pitch])))

indexes = np.argsort(timestamps)
timestamps = timestamps[indexes]
notes = notes[indexes]

fig_tm = plot_timestamps(timestamps, notes=notes)

# Piece creation
piece = Piece('Chopin', final_rest=2.)

for note in midi.instruments[0].notes:
    piece.notes.append(Note(note.pitch, 127, start_seconds=note.start, end_seconds=note.end))

# Piano roll
frequency_vector = 440 * 2**((np.arange(21., 108.) - 69.) / 12)
time_vector = np.arange(0., piece.duration(), time_resolution)
piano_roll = create_piano_roll(piece, frequency_vector, time_vector)

fig_pr = plot_piano_roll(piano_roll, frequency_vector, time_vector, show=False, block=False, fig_size=(600, 360))

# Plot Graph aCD's
pos = nx.get_node_attributes(graph, 'pos')
node_labels = {idx: round(graph.nodes[idx]['acd'], 3) for idx in graph.nodes}

color = .95 * np.ones((len(graph), 3))

fig = plt.figure(figsize=(10., 3.))
plt.axis('off')
nx.draw_networkx(graph, pos=pos, arrows=True, with_labels=True, labels=node_labels, node_color=color,
                 node_size=600, font_size=10)


plt.show()

if __name__ == '__main__':
    pass
