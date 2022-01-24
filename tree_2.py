import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from graph import create_tree_polyphonic_pruned

timestamps = np.array([0., 0.55, 1.23, 1.77, 2.4])

# Graph
graph_acds, all_nodes = create_tree_polyphonic_pruned(timestamps,
                                                      with_final_node=False, graph_width=3., graph_height=0.5)

# Plot
plt.rcParams.update({"text.usetex": True, "font.sans-serif": ["CMU Serif"]})

pos = nx.get_node_attributes(graph_acds, 'pos')
node_labels = {node.node_index: '$a_' + str(node.frame_index + 1) + '^' + str(node.frame) + '$' for node in all_nodes}
node_labels[0] = '$a^0$'
node_color = 0.9 * np.ones((len(all_nodes), 3))

fig = plt.figure(figsize=(6., 2.5))
nx.draw_networkx(graph_acds, pos=pos, arrows=True, with_labels=True, labels=node_labels, node_size=500,
                 node_color=node_color)
# nx.draw_networkx_edge_labels(graph_acds, pos, edge_labels=edge_labels)
ax = plt.gca()
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.12, 0.12])
plt.axis('off')
fig.savefig('Figure_3.eps', bbox_inches='tight', pad_inches=0, transparent=True)
plt.show()
