import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from graph import create_optimal_tree

timestamps = np.array([0., 0.55, 1.23, 1.77, 2.4])
durations = np.diff(timestamps)
print('Durations:', durations)

# Graph
graph_acds, all_nodes = create_optimal_tree(timestamps, with_final_node=True, graph_width=3., graph_height=1.)

# Shortest path
shortest_path = nx.shortest_path(graph_acds, 0, len(graph_acds.nodes) - 1)

shortest_acds = [round(all_nodes[node].duration, 2) for node in shortest_path]
shortest_bpms = [round(60/acd) for acd in shortest_acds if not acd == 0]

shortest_durations = [all_nodes[node].durations[0] for node in shortest_path if not len(all_nodes[node].durations) == 0]
shortest_durations += [all_nodes[shortest_path[-2]].durations[1]]

print('Shortest path')
print(shortest_path)
print(shortest_acds)
print(shortest_durations)
print(shortest_bpms)

# Other path
other_path = [0, 3, 6, 10, 11]

other_acds = [round(all_nodes[node].duration, 2) for node in other_path]
other_bpms = [round(60/acd) for acd in other_acds if not acd == 0]

other_durations = [all_nodes[node].durations[0] for node in other_path if not len(all_nodes[node].durations) == 0]
other_durations += [all_nodes[other_path[-2]].durations[1]]

print('Other path')
print(other_path)
print(other_acds)
print(other_durations)
print(other_bpms)

# Plot
# plt.rcParams.update({"text.usetex": True, "font.sans-serif": ["CMU Serif"]})

pos = nx.get_node_attributes(graph_acds, 'pos')
# node_labels = {node.node_index: '$a_' + str(node.frame_index + 1) + '^' + str(node.frame) + '$' for node in all_nodes}
node_labels = {node.node_index: str(round(node.duration, 2)) for node in all_nodes}
# node_labels = {node.node_index: str(node.node_index) for node in all_nodes}
node_color = 0.9 * np.ones((len(all_nodes), 3))

fig = plt.figure(figsize=(6., 2.8))
nx.draw_networkx(graph_acds, pos=pos, arrows=True, with_labels=True, labels=node_labels, node_size=500,
                 node_color=node_color, font_size=10)

edge_labels = nx.get_edge_attributes(graph_acds, 'weight')
edge_labels = {edge: np.round(edge_labels[edge], 2) for edge in edge_labels.keys()}
nx.draw_networkx_edge_labels(graph_acds, pos, edge_labels=edge_labels, font_size=8)

ax = plt.gca()
# ax.set_xlim([-1.5, 1.5])
# ax.set_ylim([-1.12, 0.12])
plt.axis('off')
fig.savefig('Figure_4.eps', bbox_inches='tight', pad_inches=0, transparent=True)
plt.show()
