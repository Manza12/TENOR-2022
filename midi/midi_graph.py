import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pretty_midi as pm
from graph import create_polyphonic_graph

# Parameters
frame_size = 1.5
hop_size = frame_size / 2

threshold = 0.05
min_cand = 0.15
max_cand = 1.
res_cand = 0.001
center = True

error_weight = 0.
tempo_var_weight = 1.

plot_steps = False

# MIDI
midi = pm.PrettyMIDI('mozart_1.mid')

timestamps = np.zeros(0)
notes = np.zeros(0)

for note in midi.instruments[0].notes:
    timestamps = np.concatenate((timestamps, np.array([note.start])))
    notes = np.concatenate((notes, np.array([note.pitch])))

indexes = np.argsort(timestamps)
timestamps = timestamps[indexes]
notes = notes[indexes]

graph = create_polyphonic_graph(timestamps, notes=notes, frame_size=frame_size, hop_size=hop_size,
                                threshold=threshold, min_cand=min_cand, max_cand=max_cand, res_cand=res_cand,
                                error_weight=error_weight, tempo_var_weight=tempo_var_weight,
                                plot_steps=plot_steps, start_node=True, final_node=True)

# Plot Graph aCD's
pos = nx.get_node_attributes(graph, 'pos')
# node_labels = {idx: array_to_string(graph.nodes[idx]['durations']) for idx in graph.nodes}
node_labels = {idx: round(graph.nodes[idx]['acd'], 3) for idx in graph.nodes}

color = .95 * np.ones((len(graph), 3))

fig = plt.figure(figsize=(6., 1.5))
plt.axis('off')
nx.draw_networkx(graph, pos=pos, arrows=True, with_labels=True, labels=node_labels, node_color=color,
                 node_size=400, font_size=8)

fig.savefig('figure_polyphonic_1.eps', bbox_inches='tight', pad_inches=0, transparent=True)
fig.savefig('figure_polyphonic_1.png', bbox_inches='tight', pad_inches=0, transparent=True)

plt.show()

if __name__ == '__main__':
    pass
