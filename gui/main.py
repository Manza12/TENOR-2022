import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog as fd

from pathlib import Path
from typing import Optional, Union, Dict, List

from matplotlib.collections import PathCollection
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Text
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import ImageTk, Image

import numpy as np
import networkx as nx

import pretty_midi as pm
from MIDISynth.music import Piece, Note
from MIDISynth.pianoroll import create_piano_roll

from utils import recover_timestamps_notes
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


########################################################################################################################


# App
class RhythmTranscriptionApp(tk.Tk):
    def __init__(self, title: str = 'Rhythm Transcription', resizable: (bool, bool) = (False, False)):
        super().__init__()

        # Title
        self.title(title)

        # Resizable
        self.resizable(resizable[0], resizable[1])

        # Models
        self.parameters_model = ParametersModel()
        self.input_model = InputModel()
        self.graph_model = GraphModel()
        self.radio_model = RadioModel()
        self.grid_model = GridModel()
        # self.transcription_model = TranscriptionModel()

        # Controllers
        self.parameters_controller = ParametersController()
        self.input_controller = InputController(self)
        self.graph_controller = GraphController(self)
        self.radio_controller = RadioController(self)
        self.grid_controller = GridController(self)
        # self.transcription_controller = TranscriptionController()

        # Views
        self.parameters_view = ParametersView(self, self.parameters_model)
        self.input_view = InputView(self, self.input_model, self.input_controller)
        self.graph_view = GraphView(self)
        self.radio_view = RadioView(self, self.radio_controller)
        self.grid_view = GridView(self, self.grid_controller)
        # self.transcription_view = TranscriptionView(self)

        # Menu
        self.menu = self.create_menu()

    def create_menu(self):
        menu = tk.Menu(master=self)
        self.config(menu=menu)

        menu.file_menu = tk.Menu(menu)
        menu.file_menu.add_command(label='Open', command=self.open_file)

        menu.add_cascade(label="File", menu=menu.file_menu, underline=0)

        return menu

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

        self.update_file_change(file_path)

    def update_file_change(self, file_path: Path):
        # Update model
        self.input_model = InputModel.from_midi_file(file_path, self.parameters_model.use_offsets)
        self.input_controller.update_current_frame()

        # Update view
        self.input_controller.update_piano_roll_view()

        # Update graph
        self.graph_controller.update_graph()


########################################################################################################################


## Parameters ##
# Model
class ParametersModel:
    # Static parameters
    threshold_from = 0.
    threshold_to = 1.

    min_candidate_from = 0.1
    max_candidate_to = 1.

    res_candidate_from = 0.001
    res_candidate_to = 0.1

    def __init__(self):
        # Parameters
        self.threshold: float = default_parameters['threshold']
        self.minimum_candidate: float = default_parameters['minimum_candidate']
        self.maximum_candidate: float = default_parameters['maximum_candidate']
        self.resolution_candidate: float = default_parameters['resolution_candidate']
        self.use_offsets: bool = default_parameters['use_offsets']
        self.frame_length: float = default_parameters['frame_length']
        self.overlap: int = default_parameters['overlap']

        self.acd_parameters: dict = {'threshold': self.threshold,
                                     'min_cand': self.minimum_candidate,
                                     'max_cand': self.maximum_candidate,
                                     'res_cand': self.resolution_candidate
                                     }


# View
class ParametersView(tk.Frame):
    # Static parameters
    pad_x = 10
    pad_y = 5

    def __init__(self, app: tk.Tk, parameters_model: ParametersModel):
        super().__init__(master=app, relief='raised', borderwidth=5)

        # Threshold widgets
        self.threshold_label: tk.Label = tk.Label(master=self, text='Threshold')
        self.threshold_label.grid(row=0, column=0, padx=self.pad_x, pady=self.pad_y)

        self.threshold_spinbox: ttk.Spinbox = ttk.Spinbox(master=self,
                                                          from_=ParametersModel.threshold_from,
                                                          to=ParametersModel.threshold_to,
                                                          increment=parameters_model.resolution_candidate)
        self.threshold_spinbox.set(parameters_model.threshold)
        self.threshold_spinbox.grid(row=0, column=1, padx=self.pad_x, pady=self.pad_y)

        # Minimum candidate widgets
        self.min_candidate_label: tk.Label = tk.Label(master=self, text='Minimum candidate')
        self.min_candidate_label.grid(row=1, column=0, padx=self.pad_x, pady=self.pad_y)

        self.min_candidate_spinbox: ttk.Spinbox = ttk.Spinbox(master=self,
                                                              from_=ParametersModel.min_candidate_from,
                                                              to=parameters_model.maximum_candidate,
                                                              increment=parameters_model.resolution_candidate)
        self.min_candidate_spinbox.set(parameters_model.minimum_candidate)
        self.min_candidate_spinbox.grid(row=1, column=1, padx=self.pad_x, pady=self.pad_y)

        # Maximum candidate widgets
        self.max_candidate_label: tk.Label = tk.Label(master=self, text='Maximum candidate')
        self.max_candidate_label.grid(row=2, column=0, padx=self.pad_x, pady=self.pad_y)

        self.max_candidate_spinbox: ttk.Spinbox = ttk.Spinbox(master=self,
                                                              from_=parameters_model.minimum_candidate,
                                                              to=ParametersModel.max_candidate_to,
                                                              increment=parameters_model.resolution_candidate)
        self.max_candidate_spinbox.set(parameters_model.maximum_candidate)
        self.max_candidate_spinbox.grid(row=2, column=1, padx=self.pad_x, pady=self.pad_y)

        # Resolution candidate widgets
        self.res_candidate_label: tk.Label = tk.Label(master=self, text='Resolution candidate')
        self.res_candidate_label.grid(row=3, column=0, padx=self.pad_x, pady=self.pad_y)

        self.res_candidate_spinbox: ttk.Spinbox = ttk.Spinbox(master=self,
                                                              from_=ParametersModel.res_candidate_from,
                                                              to=ParametersModel.res_candidate_to,
                                                              increment=parameters_model.resolution_candidate)
        self.res_candidate_spinbox.set(parameters_model.resolution_candidate)
        self.res_candidate_spinbox.grid(row=3, column=1, padx=self.pad_x, pady=self.pad_y)

        # Layout
        self.grid(row=0, column=2, padx=self.pad_x, pady=self.pad_y, sticky=tk.N)


# Controller
class ParametersController:
    def __init__(self):
        pass


## Input ##
# Model
class InputModel:
    # Static parameters
    time_resolution = 0.05

    def __init__(self):
        # Common data
        self.file_path: Optional[Union[str, Path]] = None

        self.time_start: float = 1000.
        self.min_note: int = 108
        self.max_note: int = 21
        self.duration: float = 0.

        # MIDI data
        self.midi: Optional[pm.PrettyMIDI] = None
        self.piece: Optional[Piece] = None

        self.frequency_vector: Optional[np.ndarray] = None
        self.time_vector: Optional[np.ndarray] = None
        self.piano_roll: Optional[np.ndarray] = None

        # Full data
        self.timestamps: Optional[np.ndarray] = None
        self.notes: Optional[np.ndarray] = None
        self.n: int = 0

        # Current frame data
        self.current_frame: int = 1
        self.frame_start: float = 0.
        self.frame_end: float = 0.
        self.current_timestamps: List[float] = []
        self.current_notes: List[int] = []

    @classmethod
    def from_midi_file(cls, file_path: Optional[Union[str, Path]], use_offsets: bool):
        instance = cls()
        instance.midi = pm.PrettyMIDI(open(file_path, 'rb'))
        instance.piece = Piece('', final_rest=2.)

        for instrument in instance.midi.instruments:
            for note in instrument.notes:
                instance.min_note = min(instance.min_note, note.pitch)
                instance.max_note = max(instance.max_note, note.pitch)
                instance.time_start = min(note.start, instance.time_start)
                instance.piece.notes.append(Note(note.pitch, 127, start_seconds=note.start, end_seconds=note.end))

        instance.duration = instance.piece.duration()
        instance.frequency_vector = 440 * 2 ** ((np.arange(21., 108.) - 69.) / 12)
        instance.time_vector = np.arange(0., instance.duration, instance.time_resolution)
        instance.piano_roll = create_piano_roll(instance.piece, instance.frequency_vector, instance.time_vector)

        instance.timestamps, instance.notes = recover_timestamps_notes(instance.piece, use_offsets)

        assert len(instance.timestamps) == len(instance.notes)
        instance.n = len(instance.timestamps)

        return instance


# Controller
class InputController:
    def __init__(self, app: RhythmTranscriptionApp):
        self.app: RhythmTranscriptionApp = app

    def update_view(self):
        self.app.input_view.adjust_frame(self.app.input_model)
        self.app.input_view.slider_label.configure(text='Frame ' + str(self.app.input_model.current_frame))

    def move_backward(self):
        if self.app.input_model.current_frame == 1:
            pass
        else:
            # Update input
            self.app.input_model.current_frame -= 1
            self.app.input_controller.update_current_frame()

            # Update view
            self.update_view()

            # Trigger update graph view
            self.app.graph_controller.update_current_frame()

            # Trigger update grid view
            self.app.grid_controller.update_grid()

    def move_forward(self):
        if self.app.input_model.frame_end > self.app.input_model.duration:
            pass
        else:
            # Update input
            self.app.input_model.current_frame += 1
            self.app.input_controller.update_current_frame()

            # Update view
            self.update_view()

            # Trigger update graph view
            self.app.graph_controller.update_current_frame()

    def update_current_frame(self):
        input_model = self.app.input_model
        frame_length = self.app.parameters_model.frame_length
        overlap = self.app.parameters_model.overlap

        input_model.frame_start = (input_model.current_frame - 1) * frame_length / overlap + input_model.time_start
        input_model.frame_end = input_model.frame_start + frame_length

        self.app.input_model.current_timestamps = []
        self.app.input_model.current_notes = []
        started = False
        for t, timestamp in enumerate(input_model.timestamps):
            if input_model.frame_start <= timestamp < input_model.frame_end:
                started = True
                self.app.input_model.current_timestamps.append(timestamp)
                self.app.input_model.current_notes.append(input_model.notes[t])
            else:
                if started:
                    break

    def update_piano_roll_view(self):
        # Plot Piano roll
        self.app.input_view.plot_piano_roll(self.app.input_model)

        # Adjust frame
        self.app.input_view.adjust_frame(self.app.input_model)


# View
class InputView(tk.Frame):
    width = 450
    height = 250

    dpi = 100

    pad_x = 10
    pad_y = 5

    def __init__(self, app: tk.Tk, input_model: InputModel, input_controller: InputController):
        super().__init__(master=app)

        self.figure_frame = tk.Frame(master=self)
        self.slider_frame = tk.Frame(master=self)

        self.figure_frame.grid(row=0, column=0)
        self.slider_frame.grid(row=1, column=0, padx=InputView.pad_x, pady=InputView.pad_y)

        # Create figure and canvas
        self.figure: Figure = Figure(figsize=(InputView.width / InputView.dpi, InputView.height / InputView.dpi),
                                     dpi=InputView.dpi)
        self.canvas: FigureCanvasTkAgg = FigureCanvasTkAgg(self.figure, master=self.figure_frame)

        self.canvas.get_tk_widget().pack()
        self.canvas.draw()

        # Create slider
        self.slider_label = tk.Label(master=self.slider_frame, text='Frame ' + str(input_model.current_frame))
        self.slider_label.grid(row=0, column=1)

        self.backward_img = ImageTk.PhotoImage(Image.open("backward.png"))
        self.back_button = tk.Button(master=self.slider_frame, image=self.backward_img, relief=tk.FLAT,
                                     command=input_controller.move_backward)
        self.back_button.grid(row=0, column=0)

        self.forward_img = ImageTk.PhotoImage(Image.open("forward.png"))
        self.forward_button = tk.Button(master=self.slider_frame, image=self.forward_img, relief=tk.FLAT,
                                        command=input_controller.move_forward)
        self.forward_button.grid(row=0, column=2)

        # Layout
        self.grid(row=0, column=0, padx=self.pad_x, pady=self.pad_y)

    def adjust_frame(self, input_model: InputModel):
        x_0 = (input_model.frame_start - InputModel.time_resolution) / InputModel.time_resolution
        x_1 = (input_model.frame_end + InputModel.time_resolution) / InputModel.time_resolution

        ax = self.figure.gca()
        ax.set_xlim([x_0, x_1])
        ax.set_ylim([input_model.min_note - 21 - 4, input_model.max_note - 21 + 4])
        self.canvas.draw()

    def plot_piano_roll(self, input_model: InputModel):
        ax = self.figure.gca()

        plot_piano_roll(ax, input_model.piano_roll, input_model.time_vector, input_model.frequency_vector)

        self.figure.subplots_adjust(left=0.14, bottom=0.19, right=.97, top=.95)
        self.canvas.draw()


## Graph ##
# Model
class GraphModel:
    def __init__(self):
        self.graph = None

        self.current_acds_list: List[float] = []

    def build_graph(self, input_model: InputModel, parameters_model: ParametersModel):
        self.graph = create_polyphonic_graph(input_model.timestamps,
                                             frame_size=parameters_model.frame_length,
                                             hop_size=parameters_model.frame_length / parameters_model.overlap,
                                             start_node=False,
                                             final_node=False,
                                             error_weight=1.,
                                             tempo_var_weight=1.,
                                             **parameters_model.acd_parameters
                                             )


# View
class GraphView(tk.Frame):
    width = 450
    height = 250

    dpi = 100

    pad_x = 10
    pad_y = 5

    color = 0.9

    def __init__(self, app: RhythmTranscriptionApp):
        super().__init__(master=app)
        self.master: RhythmTranscriptionApp = app

        # Create figure and canvas
        self.figure: Figure = Figure(figsize=(InputView.width / InputView.dpi, InputView.height / InputView.dpi),
                                     dpi=InputView.dpi)
        self.canvas: FigureCanvasTkAgg = FigureCanvasTkAgg(self.figure, master=self)
        self.ax: Optional[Axes] = self.figure.gca()

        self.canvas.get_tk_widget().pack()
        self.canvas.draw()

        # Create empty values
        self.color_nodes: Optional[np.ndarray] = None

        self.pos: Optional[Dict[(int, int), (float, float)]] = None
        self.node_labels: Optional[Dict[(int, int), float]] = None

        self.nodes: Optional[PathCollection] = None
        self.edges: Optional[List[FancyArrowPatch]] = None
        self.labels: Optional[Dict[(int, int), Text]] = None

        # Layout
        self.grid(row=1, column=0, padx=self.pad_x, pady=self.pad_y)

    def plot_graph(self, graph_model: GraphModel):
        graph = graph_model.graph

        self.pos = nx.get_node_attributes(graph, 'pos')
        self.node_labels = {idx: round(graph.nodes[idx]['acd'], 3) for idx in graph.nodes}

        self.color_nodes = self.color * np.ones((len(graph), 3))

        self.canvas.get_tk_widget().pack(side=tk.TOP, expand=True, fill=tk.BOTH, anchor=tk.NW)

        self.ax.remove()
        self.ax = self.figure.gca()
        self.ax.axis('off')

        # Plot graph
        self.nodes = nx.draw_networkx_nodes(graph, pos=self.pos, ax=self.ax, node_size=800, node_color=self.color_nodes)
        self.edges = nx.draw_networkx_edges(graph, pos=self.pos, ax=self.ax, arrows=True)
        self.labels = nx.draw_networkx_labels(graph, pos=self.pos, ax=self.ax, labels=self.node_labels, font_size=12)

        # Draw
        self.canvas.draw()

    def adjust_graph(self, input_model: InputModel):
        # Update horizontal limits
        center_frame = input_model.current_frame - 1
        self.ax.set_xlim(center_frame - 2 - 0.2, center_frame + 2 + 0.2)

        # Update vertical limits
        y_0 = 0
        y_1 = 0
        for key in self.pos.keys():
            y_0 = min(y_0, self.pos[key][1])
            y_1 = max(y_1, self.pos[key][1])

        # Draw
        self.canvas.draw()


# Controller
class GraphController:
    def __init__(self, app: RhythmTranscriptionApp):
        self.app: RhythmTranscriptionApp = app

    def update_color(self):
        current_value = self.app.radio_view.variable.get()
        nodes = self.app.graph_view.nodes
        labels = self.app.graph_view.labels
        current_frame = self.app.input_model.current_frame
        started = False
        for k, key in enumerate(labels.keys()):
            if key[0] == current_frame - 1:
                started = True
                if labels[key].get_text() == current_value:
                    nodes.properties()['edgecolor'][k] = [1., 0.7, 0.7, 1]
                else:
                    nodes.properties()['edgecolor'][k] = [GraphView.color, GraphView.color, GraphView.color, 1]
            else:
                if started:
                    break

        self.app.graph_view.canvas.draw()

    def update_graph(self):
        # Update graph model
        self.app.graph_model.build_graph(self.app.input_model, self.app.parameters_model)

        # Update graph view
        self.app.graph_view.plot_graph(self.app.graph_model)

        # Update current frame
        self.update_current_frame()

    def update_current_frame(self):
        # Update graph model
        self.update_acds_list()

        # Update graph view
        self.app.graph_view.adjust_graph(self.app.input_model)

        # Update radio
        self.app.radio_controller.update_radio(self.app.graph_model)

    def update_acds_list(self):
        self.app.graph_model.current_acds_list = []
        for key in self.app.graph_view.pos.keys():
            if key[0] == self.app.input_model.current_frame - 1:
                self.app.graph_model.current_acds_list.append(float(self.app.graph_view.labels[key].get_text()))


## Radio ##
# Model
class RadioModel:
    def __init__(self):
        self.acds_list: Optional[List[float]] = None
        self.current_value: Optional[float] = None

    @classmethod
    def from_list(cls, acds_list: List[float]):
        instance = cls()
        instance.acds_list = acds_list
        return instance


# Controller
class RadioController:
    def __init__(self, app: RhythmTranscriptionApp):
        self.app: RhythmTranscriptionApp = app

    def update_radio(self, graph_model: GraphModel):
        # Update model
        self.app.radio_model = RadioModel.from_list(graph_model.current_acds_list)

        # Update view
        self.app.radio_view.update_buttons(self.app.radio_model)

        # Trigger update grid
        self.app.grid_controller.update_grid()

    def update_button(self):
        # Update graph
        self.app.graph_controller.update_color()

        # Update grid
        self.app.grid_view.current_value = self.app.radio_view.variable
        self.app.grid_controller.plot_grid(float(self.app.grid_view.current_value.get()))


# View
class RadioView(tk.Frame):
    width = 450
    height = 250

    dpi = 100

    pad_x = 10
    pad_y = 5

    relief = tk.RAISED
    border_width = 5

    def __init__(self, app: RhythmTranscriptionApp, radio_controller: RadioController):
        super().__init__(master=app, height=RadioView.height, width=RadioView.width,
                         relief=RadioView.relief, borderwidth=RadioView.border_width)

        self.radio_controller: RadioController = radio_controller

        # Create style
        self.style = self.create_style()

        # Create buttons list
        self.buttons_list: List[ttk.Radiobutton] = []
        self.variable: tk.StringVar = tk.StringVar()

        # Layout
        self.grid(row=1, column=1, padx=self.pad_x, pady=self.pad_y)

    def create_style(self):
        style = ttk.Style(self)
        style.theme_use('clam')
        style.layout('CustomRadiobutton',
                     [('Radiobutton.padding',
                       {'children': [('Radiobutton.indicator', {'side': 'bottom', 'sticky': ''}),
                                     ('Radiobutton.focus', {'children': [('Radiobutton.label', {'sticky': 'nswe'})],
                                                            'side': 'bottom', 'sticky': ''})], 'sticky': 'nswe'})])

        return style

    def update_buttons(self, radio_model: RadioModel):
        for button in self.buttons_list:
            button.grid_remove()

        self.buttons_list = []

        for a, acd in enumerate(reversed(radio_model.acds_list)):
            radio_button = ttk.Radiobutton(self, text=str(acd), value=acd, variable=self.variable,
                                           style='CustomRadiobutton', command=self.radio_controller.update_button)
            radio_button.grid(row=0, column=a, padx=5, pady=5)

            self.buttons_list.append(radio_button)


## Grid ##
# Model
class GridModel:
    def __init__(self):
        self.timestamps: Optional[np.ndarray] = None
        self.notes: Optional[np.ndarray] = None

        self.n: int = 0

    @classmethod
    def from_input_model(cls, input_model: InputModel):
        instance = cls()
        instance.timestamps = input_model.current_timestamps
        instance.notes = input_model.current_notes
        assert len(input_model.current_timestamps) == len(input_model.current_notes)
        instance.n = len(input_model.current_timestamps)
        return instance


# Controller
class GridController:
    def __init__(self, app: RhythmTranscriptionApp):
        self.app: RhythmTranscriptionApp = app

    def update_grid(self):
        # Update model
        self.app.grid_model = GridModel.from_input_model(self.app.input_model)

        # Update view
        self.app.grid_view.plot_grid_timestamps(self.app.grid_model,
                                                self.app.parameters_model.threshold,
                                                self.app.input_model)

    def update_slider(self, value):
        self.plot_grid(float(value))

    def plot_grid(self, value: float):
        ax = self.app.grid_view.figure.gca()
        x_lim = ax.get_xlim()

        origin_timestamp = self.app.input_model.current_timestamps[0]
        last_timestamp = max(*self.app.input_model.current_timestamps)
        new_ticks = np.arange(origin_timestamp, last_timestamp + value, value)
        ax.set_xticks(new_ticks, minor=False)
        ax.set_xlim(x_lim)
        ax.xaxis.grid(True, which='major')

        self.app.grid_view.canvas.draw()

        self.app.grid_view.value.configure(text='{: .3f}'.format(value))


# View
class GridView(tk.Frame):
    width = 450
    height = 250

    dpi = 100

    pad_x = 10
    pad_y = 5

    slider_pad_x = 5
    slider_pad_y = 12

    dot_color = '#1f77b4'

    def __init__(self, app: RhythmTranscriptionApp, grid_controller: GridController):
        super().__init__(master=app)

        # Create frames
        self.figure_frame = tk.Frame(master=self)
        self.slider_frame = tk.Frame(master=self)

        self.figure_frame.grid(row=0, column=0)
        self.slider_frame.grid(row=1, column=0)

        # Create figure and canvas
        self.figure: Figure = Figure(figsize=(InputView.width / InputView.dpi, InputView.height / InputView.dpi),
                                     dpi=InputView.dpi)
        self.ax: Axes = self.figure.gca()
        self.canvas: FigureCanvasTkAgg = FigureCanvasTkAgg(self.figure, master=self.figure_frame)

        self.canvas.get_tk_widget().pack()
        self.canvas.draw()

        self.points: Optional[PathCollection] = None

        # Create slider
        self.current_value = tk.DoubleVar()
        self.label: ttk.Label = ttk.Label(master=self.slider_frame, text='ACD:')
        self.label.grid(row=0, column=0, padx=GridView.slider_pad_x, pady=GridView.slider_pad_y)

        self.value: ttk.Label = ttk.Label(master=self.slider_frame, text='')
        self.value.grid(row=0, column=1, padx=GridView.slider_pad_x, pady=GridView.slider_pad_y)

        self.slider: ttk.Scale = ttk.Scale(master=self.slider_frame, from_=app.parameters_model.minimum_candidate,
                                           to=app.parameters_model.maximum_candidate, orient=tk.HORIZONTAL,
                                           command=grid_controller.update_slider, variable=self.current_value)
        self.slider.grid(row=0, column=2, padx=GridView.slider_pad_x, pady=GridView.slider_pad_y)

        # Layout
        self.grid(row=0, column=1, padx=self.pad_x, pady=self.pad_y)

    def plot_grid_timestamps(self, grid_model: GridModel, threshold, input_model: InputModel):
        self.ax.remove()
        self.ax = self.figure.gca()

        self.points = self.ax.scatter(grid_model.timestamps, grid_model.notes, color=GridView.dot_color)

        # Limits
        self.ax.set_xlabel('Time (s)')
        self.ax.get_yaxis().set_ticks([])
        self.ax.set_ylim([np.min(grid_model.notes) - 2, np.max(grid_model.notes) + 2])
        self.ax.set_xlim(input_model.frame_start - InputModel.time_resolution,
                         input_model.frame_end + InputModel.time_resolution)

        # Threshold lines
        line = ()
        for i in range(grid_model.n):
            line = Line2D([grid_model.timestamps[i] - threshold,
                           grid_model.timestamps[i] + threshold],
                          [grid_model.notes[i], grid_model.notes[i]], color='r')
            self.ax.add_line(line)

        # Legend
        self.ax.legend([self.points, line], ['timestamps', 'threshold'])

        self.canvas.draw()


## Transcription ##
# Model
class TranscriptionModel:
    def __init__(self):
        pass


# View
class TranscriptionView(tk.Frame):
    pad_x = 10
    pad_y = 5

    def __init__(self, app: RhythmTranscriptionApp):
        super().__init__(master=app)

        # ACD widgets
        self.acd_label: tk.Label = tk.Label(master=self, text='ACD')
        self.acd_label.grid(row=0, column=0, padx=self.pad_x, pady=self.pad_y)

        self.acd_value_label: tk.Label = tk.Label(master=self, text='')
        self.acd_value_label.grid(row=0, column=1, padx=self.pad_x, pady=self.pad_y)

        # Layout
        self.grid(row=1, column=2, padx=self.pad_x, pady=self.pad_y, sticky=tk.N)


# Controller
class TranscriptionController:
    def __init__(self):
        pass


########################################################################################################################


if __name__ == "__main__":
    # Create the app
    rhythm_app = RhythmTranscriptionApp()

    # Choose a default MIDI input
    default_path = Path('.') / Path('..') / Path('midi') / 'mozart_1.mid'
    rhythm_app.update_file_change(default_path)

    # Main loop
    rhythm_app.mainloop()
