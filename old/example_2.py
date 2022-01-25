import numpy as np
# import matplotlib.pyplot as plt
# import networkx as nx
from graph import create_tree_polyphonic

timestamps = np.array([0., 0.55, 1.23, 1.77, 2.4])

# Graph
graph_acds, all_nodes = create_tree_polyphonic(timestamps, with_final_node=False)

# # Plot
# pos = nx.get_node_attributes(graph_acds, 'pos')
# edge_labels = nx.get_edge_attributes(graph_acds, 'weight')
# edge_labels = {edge: np.round(edge_labels[edge], 2) for edge in edge_labels.keys()}
# node_labels = {node.node_index: np.round(node.duration, 2) for node in all_nodes}
# node_color = 0.9 * np.ones((len(all_nodes), 3))
#
# plt.figure()
# nx.draw_networkx(graph_acds, pos=pos, arrows=True, with_labels=True, labels=node_labels, node_size=800,
#                  node_color=node_color)
# nx.draw_networkx_edge_labels(graph_acds, pos, edge_labels=edge_labels)
# plt.show()
