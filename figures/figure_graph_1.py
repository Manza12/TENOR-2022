import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from graph import create_graph
from time import time

# Parameters
from utils import array_to_string

frame_size = 3
threshold = 0.05
min_cand = 0.2
max_cand = 1.
res_cand = 0.001
center = True

# Timestamps
timestamps = np.array([0., 1.018, 1.531, 2.061, 2.888, 3.179, 4.286])

# Create Graph
start = time()
graph = create_graph(timestamps, threshold=threshold, min_cand=min_cand, max_cand=max_cand, res_cand=res_cand,
                     frame_size=frame_size)
print('Time to create graph: %.3f' % (time() - start))

# Plot Graph aCD's
pos = nx.get_node_attributes(graph, 'pos')
node_labels = {idx: round(graph.nodes[idx]['acd'], 3) for idx in graph.nodes}

color = 1. * np.ones((len(graph), 3))

fig = plt.figure(figsize=(6., 2.8))
plt.axis('off')
nx.draw_networkx(graph, pos=pos, arrows=True, with_labels=True, labels=node_labels,
                 node_size=300*frame_size, node_color=color, font_size=10)

fig.savefig('figure_graph_1-acd.eps', bbox_inches='tight', pad_inches=0, transparent=True)
fig.savefig('figure_graph_1-acd.png', bbox_inches='tight', pad_inches=0, transparent=True)

# Plot Graph durations
pos = nx.get_node_attributes(graph, 'pos')
node_labels = {idx: array_to_string(graph.nodes[idx]['durations']) for idx in graph.nodes}

color = 1. * np.ones((len(graph), 3))

fig = plt.figure(figsize=(6., 2.8))
plt.axis('off')
nx.draw_networkx(graph, pos=pos, arrows=True, with_labels=True, labels=node_labels,
                 node_size=300*frame_size, node_color=color, font_size=11)

fig.savefig('figure_graph_1-dur.eps', bbox_inches='tight', pad_inches=0, transparent=True)
fig.savefig('figure_graph_1-dur.png', bbox_inches='tight', pad_inches=0, transparent=True)
plt.show()

if __name__ == '__main__':
    pass
