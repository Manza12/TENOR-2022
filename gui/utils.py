import numpy as np
from MIDISynth.music import Piece


def recover_timestamps(piece: Piece, use_offset: bool) -> np.ndarray:
    timestamps_list = []
    for note in piece.notes:
        timestamps_list.append(note.start_seconds)

        if use_offset:
            timestamps_list.append(note.end_seconds)
    timestamps_list.sort()
    timestamps = np.array(timestamps_list)
    return timestamps


def array_to_string(array: list):
    result = '('
    for i, ele in enumerate(array):
        if i == 0:
            result += str(ele)
        else:
            result += ', ' + str(ele)
    result += ')'
    return result


def log_distance(x, y):
    return np.log2(np.maximum(x/y, y/x))


def linear_clipped_weight(value, maximum_value):
    if value > maximum_value:
        return np.float('inf')
    else:
        weight = value / maximum_value
        return weight

