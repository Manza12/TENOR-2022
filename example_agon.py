import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from graph import create_optimal_tree

durations = [1.00, 0.60, 0.30, 0.28, 0.26, 0.24, 0.22, 0.60, 1.00]
timestamps = np.cumsum([0.] + durations)
print(timestamps)

# Graph
graph_acds, all_nodes = create_optimal_tree(timestamps, with_final_node=True, graph_width=3., graph_height=1.,
                                            frame_duration=1.5)

# Plot
# plt.rcParams.update({"text.usetex": True, "font.sans-serif": ["CMU Serif"]})

pos = nx.get_node_attributes(graph_acds, 'pos')
# node_labels = {node.node_index: '$a_' + str(node.frame_index + 1) + '^' + str(node.frame) + '$' for node in all_nodes}
node_labels = {node.node_index: str(round(node.duration, 2)) for node in all_nodes}
node_color = 0.9 * np.ones((len(all_nodes), 3))

fig = plt.figure(figsize=(6., 2.5))
nx.draw_networkx(graph_acds, pos=pos, arrows=True, with_labels=True, labels=node_labels, node_size=300,
                 node_color=node_color, font_size=8)

# edge_labels = nx.get_edge_attributes(graph_acds, 'weight')
# edge_labels = {edge: np.round(edge_labels[edge], 2) for edge in edge_labels.keys()}
# nx.draw_networkx_edge_labels(graph_acds, pos, edge_labels=edge_labels)

ax = plt.gca()
# ax.set_xlim([-1.5, 1.5])
# ax.set_ylim([-1.12, 0.12])
plt.axis('off')
fig.savefig('Figure_4.eps', bbox_inches='tight', pad_inches=0, transparent=True)
plt.show()
