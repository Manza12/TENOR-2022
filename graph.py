import networkx as nx
from acds import compute_acds
from plot import plot_timestamps_radio
from utils import log_distance, linear_clipped_weight
import numpy as np


def check_compatibility(graph, p_idx, c_idx):
    if p_idx == 0:
        return True

    durations_p = graph.nodes[p_idx]['durations']
    durations_c = graph.nodes[c_idx]['durations']

    return np.all(durations_p[1:] == durations_c[0:len(durations_p)-1])


def check_compatibility_polyphonic(graph, p_idx, c_idx):
    if p_idx == 0:
        return True

    indexes_p = graph.nodes[p_idx]['indexes']
    indexes_c = graph.nodes[c_idx]['indexes']

    shift = 0
    for idx in indexes_p[0]:
        if idx not in indexes_c[0]:
            shift += 1
        else:
            break

    durations_p = graph.nodes[p_idx]['durations']
    durations_c = graph.nodes[c_idx]['durations']

    return np.all(durations_p[shift:] == durations_c[0:len(durations_p)-shift])


def compute_weight(graph, p_idx, c_idx, error_weight=1., tempo_var_weight=1.):
    if p_idx == 0:
        return 0.

    acd_p = graph.nodes[p_idx]['acd']
    acd_c = graph.nodes[c_idx]['acd']
    tempo_variation = log_distance(acd_p, acd_c)
    error = graph.nodes[c_idx]['error']
    weight_error = linear_clipped_weight(error, graph.graph['threshold'])

    total_weight = error_weight * weight_error + tempo_var_weight * tempo_variation
    return total_weight


def create_graph(timestamps, frame_size=3, start_node=False, final_node=False, **kwargs):
    # Initialisation
    graph = nx.DiGraph(**kwargs)
    K = np.zeros(0, dtype=np.int)
    previous_nodes = []

    # Start node
    if start_node:
        graph.add_node(0, acd=0., durations=np.zeros(0, dtype=np.int), pos=(-1, 0))
        previous_nodes.append(0)

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
                           pos=(n, - (k - (K[-1] - 1) / 2)))
            current_nodes.append((n, k))

            for i in range(len(previous_nodes)):
                compatible = check_compatibility(graph, previous_nodes[i], (n, k))

                if compatible:
                    weight = compute_weight(graph, previous_nodes[i], (n, k),
                                            kwargs['error_weight'],
                                            kwargs['tempo_var_weight'])
                    graph.add_edge(previous_nodes[i], (n, k), weight=weight, color=(0., 0., 0.))

        previous_nodes = current_nodes

    # Final node
    if final_node:
        graph.add_node(1, acd=0., durations=np.zeros(0, dtype=np.int),
                       pos=(graph.nodes[previous_nodes[0]]['pos'][0] + 1, 0))
        for i in range(len(previous_nodes)):
            graph.add_edge(previous_nodes[i], 1, weight=0., color=(0., 0., 0.))

    return graph


def create_polyphonic_graph(timestamps, notes, frame_size=2., hop_size=1., start_node=False, final_node=False,
                            plot_steps=False, **kwargs):
    # Initialisation
    graph = nx.DiGraph(**kwargs)
    K = np.zeros(0, dtype=np.int)
    previous_nodes = []

    # Start node
    if start_node:
        graph.add_node(0, acd=0., durations=np.zeros(0, dtype=np.int), pos=(-1, 0))
        previous_nodes.append(0)

    # Loop
    finish = False
    t_0 = timestamps[0]
    n = 0
    while not finish:
        # Initialize
        current_nodes = []

        # Select stretch
        indexes = np.where(np.logical_and(t_0 <= timestamps, timestamps < t_0 + frame_size))
        stretch = timestamps[indexes]

        # Compute aCD's
        acds, acds_errors, acds_multiples, acds_durations = compute_acds(stretch, **kwargs)

        if plot_steps:
            plot_timestamps_radio(stretch, acds, notes=notes[indexes], fig_size=(7., 4.))
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.scatter(timestamps[indexes], notes[indexes])
        # plt.scatter(acds[0] * acds_multiples[0, :] + timestamps[indexes].min(), notes[indexes])
        # plt.show()

        # Length
        K = np.concatenate((K, np.array([len(acds)])))

        # Add nodes
        for k in range(len(acds)):
            graph.add_node((n, k),
                           acd=acds[k],
                           error=acds_errors[k],
                           multiples=acds_multiples[k, :],
                           durations=acds_durations[k, :],
                           indexes=indexes,
                           pos=(n, - (k - (K[-1] - 1) / 2)))
            current_nodes.append((n, k))

            for i in range(len(previous_nodes)):
                compatible = check_compatibility_polyphonic(graph, previous_nodes[i], (n, k))

                if compatible:
                    weight = compute_weight(graph, previous_nodes[i], (n, k),
                                            kwargs['error_weight'],
                                            kwargs['tempo_var_weight'])
                    graph.add_edge(previous_nodes[i], (n, k), weight=weight, color=(0., 0., 0.))

        previous_nodes = current_nodes
        n += 1
        t_0 = t_0 + hop_size
        if t_0 > timestamps.max():
            finish = True

    # Final node
    if final_node:
        graph.add_node(1, acd=0., durations=np.zeros(0, dtype=np.int),
                       pos=(graph.nodes[previous_nodes[0]]['pos'][0] + 1, 0))
        for i in range(len(previous_nodes)):
            graph.add_edge(previous_nodes[i], 1, weight=0., color=(0., 0., 0.))

    return graph


def path_to_durations(graph, path, frame_size=3):
    if not frame_size == 3:
        raise NotImplementedError('Frame size should be 3')

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

    return path_durations
