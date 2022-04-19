import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog as fd

from pathlib import Path
from typing import Optional, Union, Dict, List

from matplotlib.collections import PathCollection
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Text
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import ImageTk, Image

import numpy as np
import networkx as nx

import pretty_midi as pm
from MIDISynth.music import Piece, Note
from MIDISynth.pianoroll import create_piano_roll

from utils import recover_timestamps
from plot import plot_piano_roll
from graph import create_polyphonic_graph


default_parameters = {'threshold': 0.05,
                      'minimum_candidate': 0.15,
                      'maximum_candidate': 1.,
                      'resolution_candidate': 0.001,
                      'use_offsets': False,
                      'frame_length': 2.,
                      'overlap': 2
                      }


class RhythmTranscriptionApp(tk.Tk):
    def __init__(self, title: str = 'Rhythm Transcription'):
        super().__init__()

        # Title
        self.title(title)

        # Data
        self.midi_data = MIDIData()
        self.acd_parameters = ACDParameters()
        self.full_graph = FullGraph()
        self.focus_graph = FocusGraph()
        self.frame_data = FrameData()
        self.transcription = Transcription()

        # Frames
        self.piano_roll_frame = PianoRollFrame(self)
        self.full_graph_frame = FullGraphFrame(self)
        self.grid_frame = GridFrame(self)
        self.focus_graph_frame = FocusGraphFrame(self)
        self.parameters_frame = ParametersFrame(self)
        self.transcription_frame = TranscriptionFrame(self)

        # Menu
        self.menu = tk.Menu(master=self)
        self.config(menu=self.menu)

        self.file_menu = tk.Menu(self.menu)
        self.file_menu.add_command(label='Open', command=self.open_file)

        self.menu.add_cascade(label="File", menu=self.file_menu, underline=0)

    def open_file(self):
        filetypes = (
            ('MIDI files', '*.mid'),
            ('All files', '*.*')
        )

        filename = fd.askopenfilename(
            title='Open a file',
            initialdir=Path('.') / Path('..') / Path('midi'),
            filetypes=filetypes
        )

        file_path = Path(filename)

        self.midi_data = MIDIData(file_path, self)


class ACDParameters:
    def __init__(self):
        # Parameters
        self.threshold: float = default_parameters['threshold']
        self.minimum_candidate: float = default_parameters['minimum_candidate']
        self.maximum_candidate: float = default_parameters['maximum_candidate']
        self.resolution_candidate: float = default_parameters['resolution_candidate']
        self.use_offsets: bool = default_parameters['use_offsets']
        self.frame_length: float = default_parameters['frame_length']
        self.overlap: int = default_parameters['overlap']

        self.parameters_dict: dict = {'threshold': self.threshold,
                                      'minimum_candidate': self.minimum_candidate,
                                      'maximum_candidate': self.maximum_candidate,
                                      'resolution_candidate': self.resolution_candidate,
                                      'use_offsets': self.use_offsets,
                                      'frame_length': self.frame_length,
                                      'overlap': self.overlap
                                      }


class MIDIData:
    time_resolution = 0.05

    def __init__(self, file_path: Optional[Union[str, Path]] = None, app: Optional[RhythmTranscriptionApp] = None):
        self.file_path: Optional[Union[str, Path]] = file_path

        if self.file_path is None:
            self.midi: Optional[pm.PrettyMIDI] = None
            self.piece: Optional[Piece] = None
            self.frequency_vector: Optional[np.ndarray] = None
            self.time_vector: Optional[np.ndarray] = None
            self.piano_roll: Optional[np.ndarray] = None
            self.time_start = 1000.
            self.min_note = 108
            self.max_note = 21
        else:
            # Initialize values
            self.time_start = 1000.
            self.min_note = 108
            self.max_note = 21

            # Get MIDI info
            self.open_midi_file()

            # Update piano roll frame (FRAME)
            app.piano_roll_frame.update_piano_roll_frame(self)

            # Update full graph (DATA)
            self.update_full_graph(app)

    def update_full_graph(self, app: RhythmTranscriptionApp):
        app.full_graph = FullGraph(self, app.acd_parameters, app)

    def open_midi_file(self):
        self.midi: Optional[pm.PrettyMIDI] = pm.PrettyMIDI(open(self.file_path, 'rb'))

        self.piece = Piece('Mozart Sonata 8', final_rest=2.)

        for instrument in self.midi.instruments:
            for note in instrument.notes:
                self.min_note = min(self.min_note, note.pitch)
                self.max_note = max(self.max_note, note.pitch)
                self.time_start = min(note.start, self.time_start)
                self.piece.notes.append(Note(note.pitch, 127, start_seconds=note.start, end_seconds=note.end))

        self.frequency_vector = 440 * 2 ** ((np.arange(21., 108.) - 69.) / 12)
        self.time_vector = np.arange(0., self.piece.duration(), self.time_resolution)
        self.piano_roll = create_piano_roll(self.piece, self.frequency_vector, self.time_vector)


class FullGraph:
    def __init__(self, midi_data: Optional[MIDIData] = None, acd_parameters: Optional[ACDParameters] = None,
                 app: Optional[RhythmTranscriptionApp] = None):
        self.midi_data: Optional[MIDIData] = midi_data
        self.acd_parameters: Optional[ACDParameters] = acd_parameters

        if self.midi_data is None or acd_parameters is None:
            self.timestamps: Optional[np.ndarray] = None
            self.graph: Optional[nx.DiGraph] = None
        else:
            # Create graph
            self.create_full_graph()

            # Update full graph frame (FRAME)
            app.full_graph_frame.update_full_graph_frame(self)

            # Update full graph (DATA)
            self.update_focus_graph(app)

    def update_focus_graph(self, app: RhythmTranscriptionApp):
        raise NotImplementedError('Trigger update of focus graph')

    def create_full_graph(self):
        self.timestamps = recover_timestamps(self.midi_data.piece, self.acd_parameters.use_offsets)
        self.graph = create_polyphonic_graph(self.timestamps,
                                             frame_size=self.acd_parameters.frame_length,
                                             hop_size=self.acd_parameters.frame_length / self.acd_parameters.overlap,
                                             start_node=False,
                                             final_node=False,
                                             error_weight=1.,
                                             tempo_var_weight=1.,
                                             **self.acd_parameters.parameters_dict
                                             )


class FocusGraph:
    def __init__(self, full_graph: Optional[FullGraph] = None):
        self.full_graph: Optional[FullGraph] = full_graph
        self.current_acd: Optional[float] = None

        if self.full_graph is None:
            pass
        else:
            raise NotImplementedError('Create focus graph with full graph data.')


class FrameData:
    def __init__(self, midi_data: Optional[MIDIData] = None, focus_graph: Optional[FocusGraph] = None):
        self.midi_data: Optional[MIDIData] = midi_data
        self.focus_graph: Optional[FocusGraph] = focus_graph

        if self.focus_graph is None or self.midi_data is None:
            pass
        else:
            raise NotImplementedError('Create grid with midi data and focus graph data.')


class Transcription:
    def __init__(self, focus_graph: Optional[FocusGraph] = None):
        self.focus_graph: Optional[FocusGraph] = focus_graph


# Frames
class PianoRollFrame(tk.Frame):
    width = 450
    height = 250

    pad_x = 10
    pad_y = 5

    current_frame = 1

    def __init__(self, app: RhythmTranscriptionApp):
        super().__init__(master=app)
        self.master: RhythmTranscriptionApp = app

        self.figure_frame = tk.Frame(master=self)
        self.slider_frame = tk.Frame(master=self)

        self.figure_frame.grid(row=0, column=0)
        self.slider_frame.grid(row=1, column=0)

        # Create figure and canvas
        self.figure: Figure = Figure(figsize=(self.width / 100, self.height / 100), dpi=100)
        self.canvas: FigureCanvasTkAgg = FigureCanvasTkAgg(self.figure, master=self.figure_frame)

        self.canvas.get_tk_widget().pack()
        self.canvas.draw()

        # Create slider
        self.slider_label = tk.Label(master=self.slider_frame, text='Frame ' + str(self.current_frame))
        self.slider_label.grid(row=0, column=1)

        self.backward_img = ImageTk.PhotoImage(Image.open("backward.png"))
        self.back_button = tk.Button(master=self.slider_frame, image=self.backward_img, relief=tk.FLAT,
                                     command=self.move_backward)
        self.back_button.grid(row=0, column=0)

        self.forward_img = ImageTk.PhotoImage(Image.open("forward.png"))
        self.forward_button = tk.Button(master=self.slider_frame, image=self.forward_img, relief=tk.FLAT,
                                        command=self.move_forward)
        self.forward_button.grid(row=0, column=2)

        # Layout
        self.grid(row=0, column=0, padx=self.pad_x, pady=self.pad_y)

    def update_piano_roll_frame(self, midi_data: MIDIData):
        # Plot Piano roll
        ax = self.figure.gca()

        plot_piano_roll(ax, midi_data.piano_roll, midi_data.time_vector, midi_data.frequency_vector)

        self.figure.subplots_adjust(left=0.14, bottom=0.19, right=.97, top=.95)
        self.canvas.draw()

        self.adjust_frame(midi_data, self.master.acd_parameters)

    def move_backward(self):
        if self.current_frame == 1:
            pass
        else:
            self.current_frame -= 1
            self.adjust_frame(self.master.midi_data, self.master.acd_parameters)

            self.slider_label = tk.Label(master=self.slider_frame, text='Frame ' + str(self.current_frame))
            self.slider_label.grid(row=0, column=1)

    def move_forward(self):
        self.current_frame += 1
        self.adjust_frame(self.master.midi_data, self.master.acd_parameters)

        self.slider_label = tk.Label(master=self.slider_frame, text='Frame ' + str(self.current_frame))
        self.slider_label.grid(row=0, column=1)

    def adjust_frame(self, midi_data: MIDIData, acd_parameters: ACDParameters):
        overlap = acd_parameters.overlap
        frame_length = acd_parameters.frame_length
        time_start = midi_data.time_start
        time_resolution = midi_data.time_resolution

        frame_start = (self.current_frame - 1) * frame_length / overlap + time_start
        frame_end = frame_start + frame_length

        x_0 = (frame_start - midi_data.time_resolution) / time_resolution
        x_1 = (frame_end + midi_data.time_resolution) / time_resolution

        ax = self.figure.gca()
        ax.set_xlim([x_0, x_1])
        ax.set_ylim([midi_data.min_note - 21 - 4, midi_data.max_note - 21 + 4])
        self.canvas.draw()


class FullGraphFrame(tk.Frame):
    width = 450
    height = 250

    pad_x = 10
    pad_y = 5

    color = 0.9

    def __init__(self, app: RhythmTranscriptionApp):
        super().__init__(master=app)
        self.master: RhythmTranscriptionApp = app

        # Create figure and canvas
        self.figure: Figure = Figure(figsize=(self.width / 100, self.height / 100), dpi=100)
        self.canvas: FigureCanvasTkAgg = FigureCanvasTkAgg(self.figure, master=self)

        self.canvas.get_tk_widget().pack()
        self.canvas.draw()

        # Create empty values
        self.color_circles: Optional[np.ndarray] = None

        self.pos: Optional[Dict[(int, int), (float, float)]] = None
        self.node_labels: Optional[Dict[(int, int), float]] = None

        self.nodes: Optional[PathCollection] = None
        self.edges: Optional[List[FancyArrowPatch]] = None
        self.labels: Optional[Dict[(int, int), Text]] = None

        # Layout
        self.grid(row=1, column=0, padx=self.pad_x, pady=self.pad_y)

    def update_full_graph_frame(self, full_graph: FullGraph):
        graph = full_graph.graph

        self.pos = nx.get_node_attributes(graph, 'pos')
        self.node_labels = {idx: round(graph.nodes[idx]['acd'], 3) for idx in graph.nodes}

        self.color_circles = self.color * np.ones((len(graph), 3))

        self.canvas.get_tk_widget().pack(side=tk.TOP, expand=True, fill=tk.BOTH, anchor=tk.NW)

        ax = self.figure.gca()
        ax.axis('off')
        center_frame = self.master.piano_roll_frame.current_frame - 1
        ax.set_xlim([center_frame - 2 - 0.2, center_frame + 2 + 0.2])

        # Plot graph
        self.nodes = nx.draw_networkx_nodes(graph, pos=self.pos, ax=ax, node_size=800, node_color=self.color_circles)
        self.edges = nx.draw_networkx_edges(graph, pos=self.pos, ax=ax, arrows=True)
        self.labels = nx.draw_networkx_labels(graph, pos=self.pos, ax=ax, labels=self.node_labels, font_size=12)

        # Update vertical limits
        y_lim = ax.get_ylim()
        ax.set_ylim([y_lim[0] - 0.5, y_lim[1] + 0.2])

        # Draw
        self.canvas.draw()


class GridFrame(tk.Frame):
    width = 450
    height = 250

    pad_x = 10
    pad_y = 5

    def __init__(self, app: RhythmTranscriptionApp):
        super().__init__(master=app)

        # Create figure and canvas
        self.figure: Figure = Figure(figsize=(self.width / 100, self.height / 100), dpi=100)
        self.canvas: FigureCanvasTkAgg = FigureCanvasTkAgg(self.figure, master=self)

        self.canvas.get_tk_widget().pack()
        self.canvas.draw()

        # Layout
        self.grid(row=0, column=1, padx=self.pad_x, pady=self.pad_y)


class FocusGraphFrame(tk.Frame):
    width = 450
    height = 250

    pad_x = 10
    pad_y = 5

    def __init__(self, app: RhythmTranscriptionApp):
        super().__init__(master=app)

        # Create figure and canvas
        self.figure: Figure = Figure(figsize=(self.width / 100, self.height / 100), dpi=100)
        self.canvas: FigureCanvasTkAgg = FigureCanvasTkAgg(self.figure, master=self)

        self.canvas.get_tk_widget().pack()
        self.canvas.draw()

        # Layout
        self.grid(row=1, column=1, padx=self.pad_x, pady=self.pad_y)


class ParametersFrame(tk.Frame):
    pad_x = 10
    pad_y = 5

    threshold_from = 0.
    threshold_to = 1.

    min_candidate_from = 0.1
    max_candidate_to = 1.

    res_candidate_from = 0.001
    res_candidate_to = 0.1

    def __init__(self, app: RhythmTranscriptionApp):
        super().__init__(master=app, relief='raised', borderwidth=5)
        self.acd_parameters: ACDParameters = app.acd_parameters

        # Threshold widgets
        self.threshold_label: tk.Label = tk.Label(master=self, text='Threshold')
        self.threshold_label.grid(row=0, column=0, padx=self.pad_x, pady=self.pad_y)

        self.threshold_spinbox: ttk.Spinbox = ttk.Spinbox(master=self,
                                                          from_=self.threshold_from,
                                                          to=self.threshold_to,
                                                          increment=self.acd_parameters.resolution_candidate)
        self.threshold_spinbox.set(self.acd_parameters.threshold)
        self.threshold_spinbox.grid(row=0, column=1, padx=self.pad_x, pady=self.pad_y)

        # Minimum candidate widgets
        self.min_candidate_label: tk.Label = tk.Label(master=self, text='Minimum candidate')
        self.min_candidate_label.grid(row=1, column=0, padx=self.pad_x, pady=self.pad_y)

        self.min_candidate_spinbox: ttk.Spinbox = ttk.Spinbox(master=self,
                                                              from_=self.min_candidate_from,
                                                              to=self.acd_parameters.maximum_candidate,
                                                              increment=self.acd_parameters.resolution_candidate)
        self.min_candidate_spinbox.set(self.acd_parameters.minimum_candidate)
        self.min_candidate_spinbox.grid(row=1, column=1, padx=self.pad_x, pady=self.pad_y)

        # Maximum candidate widgets
        self.max_candidate_label: tk.Label = tk.Label(master=self, text='Maximum candidate')
        self.max_candidate_label.grid(row=2, column=0, padx=self.pad_x, pady=self.pad_y)

        self.max_candidate_spinbox: ttk.Spinbox = ttk.Spinbox(master=self,
                                                              from_=self.acd_parameters.minimum_candidate,
                                                              to=self.max_candidate_to,
                                                              increment=self.acd_parameters.resolution_candidate)
        self.max_candidate_spinbox.set(self.acd_parameters.maximum_candidate)
        self.max_candidate_spinbox.grid(row=2, column=1, padx=self.pad_x, pady=self.pad_y)

        # Resolution candidate widgets
        self.res_candidate_label: tk.Label = tk.Label(master=self, text='Resolution candidate')
        self.res_candidate_label.grid(row=3, column=0, padx=self.pad_x, pady=self.pad_y)

        self.res_candidate_spinbox: ttk.Spinbox = ttk.Spinbox(master=self,
                                                              from_=self.res_candidate_from,
                                                              to=self.res_candidate_to,
                                                              increment=self.acd_parameters.resolution_candidate)
        self.res_candidate_spinbox.set(self.acd_parameters.resolution_candidate)
        self.res_candidate_spinbox.grid(row=3, column=1, padx=self.pad_x, pady=self.pad_y)

        # Layout
        self.grid(row=0, column=2, padx=self.pad_x, pady=self.pad_y, sticky=tk.N)


class TranscriptionFrame(tk.Frame):
    pad_x = 10
    pad_y = 5

    def __init__(self, app: RhythmTranscriptionApp):
        super().__init__(master=app)
        self.transcription: Transcription = app.transcription

        # ACD widgets
        self.acd_label: tk.Label = tk.Label(master=self, text='ACD')
        self.acd_label.grid(row=0, column=0, padx=self.pad_x, pady=self.pad_y)

        if self.transcription.focus_graph is None:
            current_acd: str = '0'
        elif self.transcription.focus_graph.current_acd is None:
            current_acd: str = '0'
        else:
            current_acd: str = str(self.transcription.focus_graph.current_acd)

        self.acd_value_label: tk.Label = tk.Label(master=self, text=current_acd)
        self.acd_value_label.grid(row=0, column=1, padx=self.pad_x, pady=self.pad_y)

        # Layout
        self.grid(row=1, column=2, padx=self.pad_x, pady=self.pad_y, sticky=tk.N)


if __name__ == "__main__":
    # Create the app
    rhythm_app = RhythmTranscriptionApp()

    # Choose a default MIDI input
    default_path = Path('.') / Path('..') / Path('midi') / 'mozart_1.mid'
    rhythm_app.midi_data = MIDIData(default_path, rhythm_app)

    # Main loop
    rhythm_app.mainloop()
