from parameters import *
import networkx as nx
from agcd import compute_acds, compute_acds_transposed, compute_acds_multiples


class Node:
    current_index: int = 0

    def __init__(self, frame: int, frame_index: int, duration: float, multiple: np.ndarray, error: float):
        self.frame = frame
        self.frame_index = frame_index
        self.duration = duration
        self.multiple = multiple
        self.error = error

        self.node_index = Node.current_index
        Node.current_index += 1

    def __str__(self):
        result = ''
        result += '[' + str(self.frame) + ', ' + str(self.frame_index) + ']'
        result += ' (' + str(self.node_index) + ')'

        return result


class NodePos(Node):
    def __init__(self, frame: int, frame_index: int, duration: float, multiple: np.ndarray, error: float, pos: tuple):
        super().__init__(frame, frame_index, duration, multiple, error)
        self.pos = pos


class NodePoly(NodePos):
    def __init__(self, frame: int, frame_index: int, duration: float, multiple: np.ndarray, durations:np.ndarray,
                 error: float, pos: tuple):
        super().__init__(frame, frame_index, duration, multiple, error, pos)
        self.durations = durations

    def recompute_acd(self, m):
        self.duration /= m
        self.durations *= m
        self.multiple *= m


class List(list):
    def __str__(self):
        result = '['
        for i, ele in enumerate(self):
            if i == 0:
                result += str(ele)
            else:
                result += ', ' + str(ele)
        result += ']'
        return result


def log_distance(x, y):
    return np.log2(np.maximum(x/y, y/x))


def linear_clipped_weight(value, maximum_value):
    if value > maximum_value:
        return np.float('inf')
    else:
        weight_tempo_variation = value / maximum_value
        return weight_tempo_variation


def compute_weight(p_node: Node, node: Node, ratios=(0, 0)):
    if ratios == (0, 0):
        error = node.error
        error_weight = linear_clipped_weight(error, MAXIMUM_ERROR)
        return error_weight
    else:
        tempo_variation = log_distance(node.duration / ratios[1], p_node.duration / ratios[0])
        tempo_variation_weight = linear_clipped_weight(tempo_variation,
                                                       log_distance(1., 1. + MAXIMUM_TEMPO_VARIATION/100))
        error = node.error
        error_weight = linear_clipped_weight(error, MAXIMUM_ERROR)

        total_weight = WEIGHT_ERROR * error_weight + WEIGHT_TEMPO_VARIATION * tempo_variation_weight

        return total_weight


def assert_multiples(p_node: Node, node: Node, minimum_acd=min_acd):
    n_0 = p_node.multiple[1]
    n_1 = node.multiple[0]
    lcm = np.lcm(n_0, n_1)
    m_0 = lcm // n_0
    m_1 = lcm // n_1
    a_0 = p_node.duration / m_0
    a_1 = node.duration / m_1
    if a_0 > minimum_acd and a_1 > minimum_acd:
        return True
    else:
        return False


def check_conditions_on_durations(a_0, a_1, minimum_acd):
    if a_0 > minimum_acd and a_1 > minimum_acd:
        return True
    else:
        return False


def get_ratios(durations_0, durations_1):
    ratio = durations_1[~(durations_0 == 0)] / durations_0[~(durations_0 == 0)]

    if np.all(ratio == ratio[0]):
        n_0 = durations_0[~(durations_0 == 0)][0]
        n_1 = durations_1[~(durations_0 == 0)][0]
        lcm = np.lcm(n_0, n_1)
        m_0 = lcm // n_0
        m_1 = lcm // n_1

        return m_0, m_1


def assert_durations(p_node: NodePoly, node: NodePoly, ratios, minimum_acd=min_acd):
    if not ratios:
        return False
    else:
        m_0 = ratios[0]
        m_1 = ratios[1]
        a_0 = p_node.duration / m_0
        a_1 = node.duration / m_1

        return check_conditions_on_durations(a_0, a_1, minimum_acd)


def create_graph(timestamps):
    graph_acds = nx.Graph()
    previous_nodes = List()
    all_nodes = List()

    frame = 0
    node_start = Node(frame, 0, (timestamps[1] - timestamps[0])[0], np.ones(2, dtype=np.int8), 0.)
    previous_nodes.append(node_start)
    all_nodes.append(node_start)
    graph_acds.add_node(node_start.node_index, pos=(frame, 0))

    for i in range(len(timestamps) - 2):
        frame = i + 1

        stretch = timestamps[i:i + 3, :]
        stretch = stretch - stretch[1]
        acds, acds_error, acds_durations = compute_acds(stretch)

        frame_index = 0
        current_nodes = List()
        for k, acd in enumerate(acds):
            n = len(acds)

            node = Node(frame, frame_index, acd, acds_durations[:, k], acds_error[k])
            current_nodes.append(node)
            all_nodes.append(node)
            graph_acds.add_node(node.node_index, pos=(frame, frame_index - (n - 1) / 2))

            for p_node in previous_nodes:
                multiples_ok = assert_multiples(p_node, node)

                if multiples_ok:
                    weight = compute_weight(p_node, node)
                    if weight < np.float('inf'):
                        graph_acds.add_edge(p_node.node_index, node.node_index, weight=weight)

            frame_index += 1

        previous_nodes = current_nodes

    # Final Node
    frame += 1
    final_node = Node(frame, 0, (timestamps[-1] - timestamps[-2])[0], np.ones(2, dtype=np.int8), 0.)
    all_nodes.append(final_node)
    graph_acds.add_node(final_node.node_index, pos=(frame, 0))
    for p_node in previous_nodes:
        graph_acds.add_edge(p_node.node_index, final_node.node_index, weight=1)

    return graph_acds, all_nodes


def create_tree(timestamps, with_final_node=True):
    tree_acds = nx.Graph()
    all_nodes = List()
    previous_nodes = List()

    frame = 0
    node_start = NodePos(frame, 0, (timestamps[1] - timestamps[0])[0], np.ones(2, dtype=np.int8), 0., pos=(0, -frame))
    all_nodes.append(node_start)
    previous_nodes.append(node_start)
    tree_acds.add_node(node_start.node_index, pos=node_start.pos)

    for i in range(len(timestamps) - 2):
        frame = i + 1

        stretch = timestamps[i:i + 3, 0]
        stretch = stretch - stretch[1]
        acds, acds_error, acds_durations = compute_acds_transposed(np.expand_dims(stretch, 0))

        current_nodes = List()
        for p_node in previous_nodes:
            for k, acd in enumerate(acds):
                n = len(acds)

                # Check go to node is possible
                node = NodePos(frame, k, acd, acds_durations[k, :], acds_error[k],
                               pos=(p_node.pos[0] + k - (n - 1) / 2, -frame))
                multiples_ok = assert_multiples(p_node, node)
                if multiples_ok:
                    weight = compute_weight(p_node, node)
                    if weight < np.float('inf'):
                        all_nodes.append(node)
                        current_nodes.append(node)

                        tree_acds.add_node(node.node_index, pos=node.pos)
                        tree_acds.add_edge(p_node.node_index, node.node_index, weight=weight)

        previous_nodes = current_nodes

    # Final Node
    if with_final_node:
        frame += 1
        final_node = NodePos(frame, 0, (timestamps[-1] - timestamps[-2])[0], np.ones(2, dtype=np.int8), 0.,
                             pos=(0, -frame))
        all_nodes.append(final_node)
        tree_acds.add_node(final_node.node_index, pos=final_node.pos)
        for p_node in previous_nodes:
            tree_acds.add_edge(p_node.node_index, final_node.node_index, weight=0)

    return tree_acds, all_nodes


def create_tree_polyphonic(timestamps, notes, frame_duration=2., with_final_node=True):
    # Initialisation
    tree_acds = nx.Graph()
    all_nodes = List()
    previous_nodes = List()

    finish = False
    starting_index = 0

    # Create first node
    node_start = NodePoly(0, 0, 0., np.zeros(0, dtype=np.int8), np.zeros(0, dtype=np.int8), 0., (0, 0))
    all_nodes.append(node_start)
    previous_nodes.append(node_start)
    tree_acds.add_node(node_start.node_index, pos=node_start.pos)

    # Loop
    while not finish:
        # Select stretch
        index = starting_index
        t_0 = timestamps[index]
        t_1 = t_0 + frame_duration
        stretch = [timestamps[index]]
        while stretch[-1] < t_1:
            if index + 1 < len(timestamps):
                if timestamps[index + 1] < t_1:
                    stretch.append(timestamps[index + 1])
                    index += 1
                else:
                    break
            else:
                finish = True
                break

        # Compute aCD's
        acds, acds_error, acds_multiples = compute_acds_multiples(np.expand_dims(np.array(stretch), 0),
                                                                  notes=notes, plot=False)

        if len(acds) == 0:
            raise NotImplementedError('Graph break not handled yet.')

        acds_durations = np.diff(acds_multiples, axis=-1)

        # Add nodes
        current_nodes = List()
        for p_node in previous_nodes:
            for k, acd in enumerate(acds):
                n = len(acds)

                # Check go to node is possible
                node = NodePoly(starting_index + 1, k, acd, acds_multiples[k, :], acds_durations[k, :], acds_error[k],
                                (p_node.pos[0] + k - (n - 1) / 2, - (starting_index + 1)))

                if p_node.durations.shape[0] == 0:
                    durations_ok = True
                    weight = compute_weight(p_node, node)
                else:
                    intersection_length = min(len(p_node.durations) - 1, len(node.durations))
                    common_durations_0 = p_node.durations[1: 1+intersection_length]
                    common_durations_1 = node.durations[: intersection_length]
                    ratios = get_ratios(common_durations_0, common_durations_1)
                    durations_ok = assert_durations(p_node, node, ratios)
                    if durations_ok:
                        weight = compute_weight(p_node, node, ratios)

                        # Recalculate node durations
                        node.recompute_acd(ratios[1])
                    else:
                        weight = np.float('inf')

                if durations_ok:
                    if weight < np.float('inf'):
                        # Add node to structures
                        all_nodes.append(node)
                        current_nodes.append(node)

                        tree_acds.add_node(node.node_index, pos=node.pos)
                        tree_acds.add_edge(p_node.node_index, node.node_index, weight=weight)
                    else:
                        Node.current_index -= 1
                else:
                    Node.current_index -= 1

        previous_nodes = current_nodes
        starting_index += 1

    # Final Node
    if with_final_node:
        final_node = NodePoly(starting_index + 1, 0, 0., np.ones(0, dtype=np.int8), np.ones(0, dtype=np.int8), 0.,
                              (0, -(starting_index + 1)))
        all_nodes.append(final_node)
        tree_acds.add_node(final_node.node_index, pos=final_node.pos)
        for p_node in previous_nodes:
            tree_acds.add_edge(p_node.node_index, final_node.node_index, weight=0)

    return tree_acds, all_nodes
