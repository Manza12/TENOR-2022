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

error_weight = 0.
tempo_var_weight = 1.

color_edges = False

# Timestamps
timestamps = np.array([0., 1.018, 1.531, 2.061, 2.888, 3.179, 4.286])

# Create Graph
start = time()
graph = create_graph(timestamps, threshold=threshold, min_cand=min_cand, max_cand=max_cand, res_cand=res_cand,
                     frame_size=frame_size, error_weight=error_weight, tempo_var_weight=tempo_var_weight)
print('Time to create graph: %.3f' % (time() - start))

# Shortest path
# path = nx.shortest_path(graph, source=0, target=1)

# Plot Graph aCD's
pos = nx.get_node_attributes(graph, 'pos')
node_labels = {idx: round(graph.nodes[idx]['acd'], 3) for idx in graph.nodes}

edge_labels = nx.get_edge_attributes(graph, 'weight')
edge_labels = {edge: np.round(edge_labels[edge], 2) for edge in edge_labels.keys()}

for key in edge_labels.keys():
    print(key, edge_labels[key])


if __name__ == '__main__':
    pass
