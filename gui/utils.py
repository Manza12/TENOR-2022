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


def recover_timestamps_notes(piece: Piece, use_offset: bool) -> (np.ndarray, np.ndarray):
    timestamps_notes_list = []
    for note in piece.notes:
        timestamps_notes_list.append((note.start_seconds, note.note_number))

        if use_offset:
            timestamps_notes_list.append((note.end_seconds, note.note_number))
    timestamps_notes_list.sort(key=lambda pair: pair[0])
    timestamps_notes = np.array(timestamps_notes_list)
    return timestamps_notes[:, 0], timestamps_notes[:, 1]


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

