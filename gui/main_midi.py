import tkinter as tk
from tkinter import filedialog as fd

from pathlib import Path
from typing import Optional, Union

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import ImageTk, Image

import numpy as np

import pretty_midi as pm
from MIDISynth.music import Piece, Note
from MIDISynth.pianoroll import create_piano_roll

from plot import plot_piano_roll

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
            self.time_start = 1000.
            self.min_note = 108
            self.max_note = 21
            self.open_midi_file(app)
            raise NotImplementedError('Update full graph with MIDI data.')

    def open_midi_file(self, app: RhythmTranscriptionApp):
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

        # Plot Piano roll
        ax = app.piano_roll_frame.figure.gca()

        plot_piano_roll(ax, self.piano_roll, self.time_vector, self.frequency_vector)

        app.piano_roll_frame.figure.subplots_adjust(left=0.14, bottom=0.19, right=.97, top=.95)
        app.piano_roll_frame.canvas.draw()

        app.piano_roll_frame.adjust_frame(app)


class FullGraph:
    def __init__(self, midi_data: Optional[MIDIData] = None, acd_parameters: Optional[ACDParameters] = None):
        self.midi_data: Optional[MIDIData] = midi_data
        self.acd_parameters: Optional[ACDParameters] = acd_parameters

        if self.midi_data is None or acd_parameters is None:
            pass
        else:
            raise NotImplementedError('Create full graph with midi data and acd parameters.')


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

    def move_backward(self):
        if self.current_frame == 1:
            pass
        else:
            self.current_frame -= 1
            self.adjust_frame(self.master)

            self.slider_label = tk.Label(master=self.slider_frame, text='Frame ' + str(self.current_frame))
            self.slider_label.grid(row=0, column=1)

    def move_forward(self):
        self.current_frame += 1
        self.adjust_frame(self.master)

        self.slider_label = tk.Label(master=self.slider_frame, text='Frame ' + str(self.current_frame))
        self.slider_label.grid(row=0, column=1)

    def adjust_frame(self, app: RhythmTranscriptionApp):
        overlap = app.acd_parameters.overlap
        frame_length = app.acd_parameters.frame_length
        time_start = app.midi_data.time_start
        time_resolution = app.midi_data.time_resolution

        frame_start = (self.current_frame - 1) * frame_length / overlap + time_start
        frame_end = frame_start + frame_length

        x_0 = (frame_start - app.midi_data.time_resolution) / time_resolution
        x_1 = (frame_end + app.midi_data.time_resolution) / time_resolution

        ax = self.figure.gca()
        ax.set_xlim([x_0, x_1])
        ax.set_ylim([app.midi_data.min_note - 21 - 4, app.midi_data.max_note - 21 + 4])
        self.canvas.draw()


class FullGraphFrame(tk.Frame):
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
        self.grid(row=1, column=0, padx=self.pad_x, pady=self.pad_y)


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

        self.threshold_spinbox: tk.Spinbox = tk.Spinbox(master=self,
                                                        from_=self.threshold_from,
                                                        to=self.threshold_to,
                                                        increment=self.acd_parameters.resolution_candidate)
        self.threshold_spinbox.grid(row=0, column=1, padx=self.pad_x, pady=self.pad_y)

        # Minimum candidate widgets
        self.min_candidate_label: tk.Label = tk.Label(master=self, text='Minimum candidate')
        self.min_candidate_label.grid(row=1, column=0, padx=self.pad_x, pady=self.pad_y)

        self.min_candidate_spinbox: tk.Spinbox = tk.Spinbox(master=self,
                                                            from_=self.min_candidate_from,
                                                            to=self.acd_parameters.maximum_candidate,
                                                            increment=self.acd_parameters.resolution_candidate)
        self.min_candidate_spinbox.grid(row=1, column=1, padx=self.pad_x, pady=self.pad_y)

        # Maximum candidate widgets
        self.max_candidate_label: tk.Label = tk.Label(master=self, text='Maximum candidate')
        self.max_candidate_label.grid(row=2, column=0, padx=self.pad_x, pady=self.pad_y)

        self.max_candidate_spinbox: tk.Spinbox = tk.Spinbox(master=self,
                                                            from_=self.acd_parameters.minimum_candidate,
                                                            to=self.max_candidate_to,
                                                            increment=self.acd_parameters.resolution_candidate)
        self.max_candidate_spinbox.grid(row=2, column=1, padx=self.pad_x, pady=self.pad_y)

        # Resolution candidate widgets
        self.res_candidate_label: tk.Label = tk.Label(master=self, text='Resolution candidate')
        self.res_candidate_label.grid(row=3, column=0, padx=self.pad_x, pady=self.pad_y)

        self.res_candidate_spinbox: tk.Spinbox = tk.Spinbox(master=self,
                                                            from_=self.res_candidate_from,
                                                            to=self.res_candidate_to,
                                                            increment=self.acd_parameters.resolution_candidate)
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
    rhythm_app.midi_data.file_path = Path('.') / Path('..') / Path('midi') / 'mozart_1.mid'
    rhythm_app.midi_data.open_midi_file(rhythm_app)

    # Main loop
    rhythm_app.mainloop()
