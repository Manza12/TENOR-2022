import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from utils import array_to_string
from graph import create_graph
from time import time

# Parameters
frame_size = 3
threshold = 0.05
min_cand = 0.2
max_cand = 1.
res_cand = 0.001
center = True

error_weight = 1.
tempo_var_weight = 1.

# Timestamps
timestamps = np.array([0., 1.018, 1.531, 2.061, 2.888, 3.179, 4.286])

# Create Graph
start = time()
graph = create_graph(timestamps, threshold=threshold, min_cand=min_cand, max_cand=max_cand, res_cand=res_cand,
                     frame_size=frame_size, start_node=True, final_node=True,
                     error_weight=error_weight, tempo_var_weight=tempo_var_weight)
print('Time to create graph: %.3f' % (time() - start))

# Shortest path
path = nx.shortest_path(graph, source=0, target=1, weight='weight')
print(path)
path_durations = np.zeros(0, dtype=np.int)
for i, idx in enumerate(path):
    node = graph.nodes[idx]
    durations = node['durations']

    if i == 0:
        pass
    if i == 1:
        path_durations = np.concatenate((path_durations, durations))
    else:
        path_durations = np.concatenate((path_durations, durations[1:]))

print(path_durations)

# Plot Graph aCD's
color_nodes = False
color_edges = True

pos = nx.get_node_attributes(graph, 'pos')
node_labels = {idx: array_to_string(graph.nodes[idx]['durations']) for idx in graph.nodes}

edge_labels = nx.get_edge_attributes(graph, 'weight')
edge_labels = {edge: np.round(edge_labels[edge], 2) for edge in edge_labels.keys()}

color_n = 1. * np.ones((len(graph), 3))
if color_nodes:
    for i, node in enumerate(graph.nodes):
        for ele in path:
            if ele == node:
                color_n[i, :] = (1., 0., 0.)

if color_edges:
    color_e = []
    for u, v in graph.edges():
        if u in path and v in path:
            color_e.append((1., 0., 0.))
        else:
            color_e.append(graph[u][v]['color'])
else:
    color_e = [graph[u][v]['color'] for u, v in graph.edges()]

fig = plt.figure(figsize=(6., 2.8))
plt.axis('off')
nx.draw_networkx(graph, pos=pos, arrows=True, with_labels=True, labels=node_labels,
                 node_size=600, node_color=color_n, edge_color=color_e, font_size=10)
nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
plt.show()

if __name__ == '__main__':
    pass
