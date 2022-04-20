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
        # self.grid_model = GridModel()
        # self.transcription_model = TranscriptionModel()

        # Controllers
        self.parameters_controller = ParametersController()
        self.input_controller = InputController(self)
        self.graph_controller = GraphController(self)
        self.radio_controller = RadioController(self)
        # self.grid_controller = GridController()
        # self.transcription_controller = TranscriptionController()

        # Views
        self.parameters_view = ParametersView(self, self.parameters_model)
        self.input_view = InputView(self, self.input_model, self.input_controller)
        self.graph_view = GraphView(self)
        self.radio_view = RadioView(self)
        # self.grid_view = GridView(self)
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
        self.input_model = InputModel.from_midi_file(file_path)

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
        self.current_frame: int = 1
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

    @classmethod
    def from_midi_file(cls, file_path: Optional[Union[str, Path]]):
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

        return instance


# Controller
class InputController:
    def __init__(self, app: RhythmTranscriptionApp):
        self.app: RhythmTranscriptionApp = app

    def move_backward(self):
        if self.app.input_model.current_frame == 1:
            pass
        else:
            self.app.input_model.current_frame -= 1
            self.app.input_view.adjust_frame(self.app.input_model, self.app.parameters_model)

            self.app.input_view.slider_label.grid_remove()
            self.app.input_view.slider_label = tk.Label(master=self.app.input_view.slider_frame,
                                                        text='Frame ' + str(self.app.input_model.current_frame))
            self.app.input_view.slider_label.grid(row=0, column=1)

            # Trigger update frame
            self.app.graph_controller.update_current_frame()

    def move_forward(self):
        start = self.app.input_model.time_start
        hop_size = self.app.parameters_model.frame_length / self.app.parameters_model.overlap
        time_end_frame = self.app.input_model.current_frame * hop_size + start + self.app.parameters_model.frame_length
        if time_end_frame > self.app.input_model.duration:
            pass
        else:
            self.app.input_model.current_frame += 1
            self.app.input_view.adjust_frame(self.app.input_model, self.app.parameters_model)

            self.app.input_view.slider_label.grid_remove()
            self.app.input_view.slider_label = tk.Label(master=self.app.input_view.slider_frame,
                                                        text='Frame ' + str(self.app.input_model.current_frame))
            self.app.input_view.slider_label.grid(row=0, column=1)

            # Trigger update frame
            self.app.graph_controller.update_current_frame()

    def update_piano_roll_view(self):
        # Plot Piano roll
        self.app.input_view.plot_piano_roll(self.app.input_model)

        # Adjust frame
        self.app.input_view.adjust_frame(self.app.input_model, self.app.parameters_model)

    # def update_full_graph(self, ):
    #     app.graph_model = GraphModel(self, app.parameters_model, app)
    #


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
        self.slider_frame.grid(row=1, column=0)

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

    def adjust_frame(self, input_model: InputModel, parameters_model: ParametersModel):
        frame_start = (input_model.current_frame - 1) * parameters_model.frame_length / parameters_model.overlap
        frame_start += input_model.time_start
        frame_end = frame_start + parameters_model.frame_length

        x_0 = (frame_start - InputModel.time_resolution) / InputModel.time_resolution
        x_1 = (frame_end + InputModel.time_resolution) / InputModel.time_resolution

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
        self.timestamps = None
        self.graph = None

        self.current_frame: int = 1
        self.current_acds_list: List[float] = []

    def build_graph(self, input_model: InputModel, parameters_model: ParametersModel):
        self.timestamps = recover_timestamps(input_model.piece, parameters_model.use_offsets)
        self.graph = create_polyphonic_graph(self.timestamps,
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
        self.ax: Optional[Axes] = None

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

        self.ax = self.figure.gca()
        self.ax.axis('off')

        # Plot graph
        self.nodes = nx.draw_networkx_nodes(graph, pos=self.pos, ax=self.ax, node_size=800, node_color=self.color_nodes)
        self.edges = nx.draw_networkx_edges(graph, pos=self.pos, ax=self.ax, arrows=True)
        self.labels = nx.draw_networkx_labels(graph, pos=self.pos, ax=self.ax, labels=self.node_labels, font_size=12)

        # Draw
        self.canvas.draw()

    def adjust_graph(self, graph_model: GraphModel):
        # Update horizontal limits
        center_frame = graph_model.current_frame - 1
        self.ax.set_xlim(center_frame - 2 - 0.2, center_frame + 2 + 0.2)

        # Update vertical limits
        graph_model.current_acds_list = []
        y_0 = 0
        y_1 = 0
        for key in self.pos.keys():
            if key[0] == center_frame:
                graph_model.current_acds_list.append(float(self.labels[key].get_text()))
                y_0 = min(y_0, self.pos[key][1])
                y_1 = max(y_1, self.pos[key][1])
        self.ax.set_ylim(y_0 - 0.5, y_1 + 0.5)

        # Draw
        self.canvas.draw()


# Controller
class GraphController:
    def __init__(self, app: RhythmTranscriptionApp):
        self.app: RhythmTranscriptionApp = app

    def update_graph(self):
        # Update frame
        self.app.graph_model.current_frame = self.app.input_model.current_frame

        # Update graph model
        self.app.graph_model.build_graph(self.app.input_model, self.app.parameters_model)

        # Update graph view
        self.app.graph_view.plot_graph(self.app.graph_model)
        self.app.graph_view.adjust_graph(self.app.graph_model)

        # Trigger radio
        self.app.radio_controller.update_radio(self.app.graph_model)

    def update_current_frame(self):
        # Update frame
        self.app.graph_model.current_frame = self.app.input_model.current_frame

        # Update graph
        self.app.graph_view.adjust_graph(self.app.graph_model)

        # Update radio
        self.app.radio_controller.update_radio(self.app.graph_model)


## Radio ##
# Model
class RadioModel:
    def __init__(self):
        self.acds_list: Optional[List[float]] = None

    @classmethod
    def from_list(cls, acds_list: List[float]):
        instance = cls()
        instance.acds_list = acds_list
        return instance


# View
class RadioView(tk.Frame):
    width = 450
    height = 250

    dpi = 100

    pad_x = 10
    pad_y = 5

    relief = tk.RAISED
    border_width = 5

    def __init__(self, app: RhythmTranscriptionApp):
        super().__init__(master=app, height=RadioView.height, width=RadioView.width,
                         relief=RadioView.relief, borderwidth=RadioView.border_width)

        # Create buttons list
        self.buttons_list: List[ttk.Radiobutton] = []
        self.variable: tk.StringVar = tk.StringVar()

        # Layout
        self.grid(row=1, column=1, padx=self.pad_x, pady=self.pad_y)

    def update_buttons(self, radio_model: RadioModel):
        for button in self.buttons_list:
            button.grid_remove()

        for a, acd in enumerate(radio_model.acds_list):
            radio_button = ttk.Radiobutton(self, text=str(acd), value=acd, variable=self.variable)
            radio_button.grid(row=0, column=a, padx=5, pady=5)

            self.buttons_list.append(radio_button)


# Controller
class RadioController:
    def __init__(self, app: RhythmTranscriptionApp):
        self.app: RhythmTranscriptionApp = app

    def update_radio(self, graph_model: GraphModel):
        # Update model
        self.app.radio_model = RadioModel.from_list(graph_model.current_acds_list)

        # Update view
        self.app.radio_view.update_buttons(self.app.radio_model)


# ## Grid ##
# # Model
# class GridModel:
#     def __init__(self, midi_data: Optional[MIDIData] = None, focus_graph: Optional[FocusGraph] = None):
#         self.midi_data: Optional[MIDIData] = midi_data
#         self.focus_graph: Optional[FocusGraph] = focus_graph
#
#         if self.focus_graph is None or self.midi_data is None:
#             pass
#         else:
#             raise NotImplementedError('Create grid with midi data and focus graph data.')

#
# # View
# class GridView(tk.Frame):
#     width = 450
#     height = 250
#
#     pad_x = 10
#     pad_y = 5
#
#     def __init__(self, app: RhythmTranscriptionApp):
#         super().__init__(master=app)
#
#         # Create figure and canvas
#         self.figure: Figure = Figure(figsize=(self.width / 100, self.height / 100), dpi=100)
#         self.canvas: FigureCanvasTkAgg = FigureCanvasTkAgg(self.figure, master=self)
#
#         self.canvas.get_tk_widget().pack()
#         self.canvas.draw()
#
#         # Layout
#         self.grid(row=0, column=1, padx=self.pad_x, pady=self.pad_y)
#
#
# # Controller
# class GridController:
#     def __init__(self):
#         pass
#
#
# ## Transcription ##
# # Model
# class TranscriptionModel:
#     def __init__(self, focus_graph: Optional[FocusGraph] = None):
#         self.focus_graph: Optional[FocusGraph] = focus_graph
#
#
# # View
# class TranscriptionView(tk.Frame):
#     pad_x = 10
#     pad_y = 5
#
#     def __init__(self, app: RhythmTranscriptionApp):
#         super().__init__(master=app)
#         self.transcription: TranscriptionModel = app.transcription_model
#
#         # ACD widgets
#         self.acd_label: tk.Label = tk.Label(master=self, text='ACD')
#         self.acd_label.grid(row=0, column=0, padx=self.pad_x, pady=self.pad_y)
#
#         if self.transcription.focus_graph is None:
#             current_acd: str = '0'
#         elif self.transcription.focus_graph.current_acd is None:
#             current_acd: str = '0'
#         else:
#             current_acd: str = str(self.transcription.focus_graph.current_acd)
#
#         self.acd_value_label: tk.Label = tk.Label(master=self, text=current_acd)
#         self.acd_value_label.grid(row=0, column=1, padx=self.pad_x, pady=self.pad_y)
#
#         # Layout
#         self.grid(row=1, column=2, padx=self.pad_x, pady=self.pad_y, sticky=tk.N)
#
#
# # Controller
# class TranscriptionController:
#     def __init__(self):
#         pass


########################################################################################################################


if __name__ == "__main__":
    # Create the app
    rhythm_app = RhythmTranscriptionApp()

    # Choose a default MIDI input
    default_path = Path('.') / Path('..') / Path('midi') / 'chopin_1.mid'
    rhythm_app.update_file_change(default_path)

    # Main loop
    rhythm_app.mainloop()
