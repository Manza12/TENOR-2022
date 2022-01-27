import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from graph import create_graph, path_to_durations
from time import time

# Parameters
frame_size = 3
threshold = 0.05
min_cand = 0.2
max_cand = 1.
res_cand = 0.001
center = True

error_weight = 0.
tempo_var_weight = 1.

color_edges = True

# Timestamps
timestamps = np.array([0., 1.018, 1.531, 2.061, 2.888, 3.179, 4.286])

# Create Graph
start = time()
graph = create_graph(timestamps, threshold=threshold, min_cand=min_cand, max_cand=max_cand, res_cand=res_cand,
                     frame_size=frame_size, error_weight=error_weight, tempo_var_weight=tempo_var_weight,
                     final_node=True, start_node=True)
print('Time to create graph: %.3f' % (time() - start))

# Shortest path
path = nx.shortest_path(graph, source=0, target=1, weight='weight')
path_durations = path_to_durations(graph, path, frame_size=frame_size)
print(path_durations)

# Plot Graph aCD's
pos = nx.get_node_attributes(graph, 'pos')
node_labels = {idx: round(graph.nodes[idx]['acd'], 3) for idx in graph.nodes}

edge_labels = nx.get_edge_attributes(graph, 'weight')
edge_labels = {edge: np.round(edge_labels[edge], 2) for edge in edge_labels.keys()}

color = 1. * np.ones((len(graph), 3))

if color_edges:
    color_e = []
    for u, v in graph.edges():
        if u in path and v in path:
            color_e.append((1., 0., 0.))
        else:
            color_e.append(graph[u][v]['color'])
else:
    color_e = [graph[u][v]['color'] for u, v in graph.edges()]

fig = plt.figure(figsize=(6., 1.5))
plt.axis('off')
nx.draw_networkx(graph, pos=pos, arrows=True, with_labels=True, labels=node_labels,
                 node_size=300*frame_size, node_color=color, edge_color=color_e, font_size=12)
# nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=10)

fig.savefig('figure_graph_3.eps', bbox_inches='tight', pad_inches=0, transparent=True)
fig.savefig('figure_graph_3.png', bbox_inches='tight', pad_inches=0, transparent=True)

plt.show()

if __name__ == '__main__':
    pass
