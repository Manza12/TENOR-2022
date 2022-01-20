import numpy as np

from parameters import *
from graph import create_tree_polyphonic, get_ratios
import networkx as nx
import matplotlib.pyplot as plt

# Input
notes = [60, 60, 67, 64, 65, 64, 62, 65, 67, 60]
print(len(notes))
timestamps = [0, 0.965, 0.973, 0.986, 1.93, 2.53, 3.023, 3.06, 3.09, 4.1]
timestamps = np.array(timestamps)
print(timestamps)
print(len(timestamps))

# Graph
graph_acds, all_nodes = create_tree_polyphonic(timestamps, notes, with_final_node=True)

# Shortest path
shortest_path = nx.shortest_path(graph_acds, 0, len(graph_acds.nodes) - 1)
print(shortest_path)

durations = np.array([], dtype=np.int8)
acds = np.array([])
for i, node_index in enumerate(shortest_path):
    # print(i, node_index, len(acds), len(durations))
    if i == 0:
        pass
    else:
        node = all_nodes[node_index]
        if i == 1:
            durations = np.concatenate((durations, node.durations))
            acds = np.concatenate((acds, node.duration * np.ones(1)))
        else:
            if not len(node.durations) == 0:
                intersection_length = min(len(durations) - (i - 1), len(node.durations))
                common_durations_0 = durations[i - 1: i - 1 + intersection_length]
                common_durations_1 = node.durations[: intersection_length]
                ratios = get_ratios(common_durations_0, common_durations_1)
                assert ratios[1] == 1

                durations *= ratios[0]
                durations = np.concatenate((durations, node.durations[intersection_length:]))

                acds /= ratios[0]
                acds = np.concatenate((acds, node.duration * np.ones(1)))
            else:
                break

print(durations)
print(len(durations))

multiples = np.cumsum(np.concatenate((np.array([0], dtype=np.int8), durations)))
print(multiples)
print(len(multiples))

# Plot
pos = nx.get_node_attributes(graph_acds, 'pos')
edge_labels = nx.get_edge_attributes(graph_acds, 'weight')
edge_labels = {edge: np.round(edge_labels[edge], 2) for edge in edge_labels.keys()}
node_labels = {node.node_index: np.round(node.duration, 2) for node in all_nodes}
node_color = 0.9 * np.ones((len(all_nodes), 3))

plt.figure()
nx.draw_networkx(graph_acds, pos=pos, arrows=True, with_labels=True, labels=node_labels, node_size=800,
                 node_color=node_color)
nx.draw_networkx_edge_labels(graph_acds, pos, edge_labels=edge_labels)
plt.show()
