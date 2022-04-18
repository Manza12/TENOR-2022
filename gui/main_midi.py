import tkinter as tk
from tkinter import filedialog as fd

from pathlib import Path
from typing import Optional, Union

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


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

        self.midi_data = MIDIData(file_path)


class MIDIData:
    def __init__(self, file_path: Optional[Union[str, Path]] = None):
        self.file_path: Optional[Union[str, Path]] = file_path

        if self.file_path is None:
            pass
        else:
            raise NotImplementedError('Update full graph with MIDI data.')


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
            raise NotImplementedError('Create grid with midi data and docus graph data.')


class Transcription:
    def __init__(self, focus_graph: Optional[FocusGraph] = None):
        self.focus_graph: Optional[FocusGraph] = focus_graph


# Frames
class PianoRollFrame(tk.Frame):
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
        self.grid(row=0, column=0, padx=self.pad_x, pady=self.pad_y)


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
    rhythm_app = RhythmTranscriptionApp()
    rhythm_app.mainloop()
