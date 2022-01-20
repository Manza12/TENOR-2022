from graph import create_graph
from parameters import *
import networkx as nx
import matplotlib.pyplot as plt

# Inputs
durations = [1.00, 0.60, 0.30, 0.28, 0.26, 0.24, 0.22, 0.60, 1.00]
timestamps = np.cumsum([0.] + durations)
timestamps = np.expand_dims(timestamps, 1)

# Graph
graph_acds, all_nodes = create_graph(timestamps)

# Check connected components
is_connected = nx.is_connected(graph_acds)
if not is_connected:
    connected_components = [c for c in nx.connected_components(graph_acds)]
else:
    # Shortest path
    shortest_path = nx.shortest_path(graph_acds, 0, len(graph_acds.nodes) - 1)
    print(shortest_path)

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
