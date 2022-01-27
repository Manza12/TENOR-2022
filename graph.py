import networkx as nx
from acds import compute_acds
import numpy as np


def check_compatibility(graph, p_idx, c_idx):
    durations_p = graph.nodes[p_idx]['durations']
    durations_c = graph.nodes[c_idx]['durations']

    return np.all(durations_p[1:] == durations_c[0:len(durations_p)-1])


def compute_weight():
    return 0.


def create_graph(timestamps, frame_size=3, **kwargs):
    # Initialisation
    graph = nx.DiGraph()
    K = np.zeros(0, dtype=np.int)
    previous_nodes = []

    # Loop
    for n in range(len(timestamps) - frame_size + 1):
        # Initialize
        current_nodes = []

        # Select stretch
        stretch = timestamps[n:n + frame_size]

        # Compute aCD's
        acds, acds_errors, acds_multiples, acds_durations = compute_acds(stretch, **kwargs)

        # Length
        K = np.concatenate((K, np.array([len(acds)])))

        # Add nodes
        for k in range(len(acds)):
            graph.add_node((n, k),
                           acd=acds[k],
                           error=acds_errors[k],
                           multiples=acds_multiples[k, :],
                           durations=acds_durations[k, :],
                           pos=(n, (k - (K[-1] - 1) / 2)))
            current_nodes.append((n, k))

            for i in range(len(previous_nodes)):
                compatible = check_compatibility(graph, previous_nodes[i], (n, k))

                if compatible:
                    weight = compute_weight()
                    graph.add_edge(previous_nodes[i], (n, k), weight=weight)

        previous_nodes = current_nodes

    return graph
