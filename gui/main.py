import tkinter as tk
import tkinter.ttk as ttk

from matplotlib.figure import Figure
from matplotlib.widgets import Slider
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.lines as lines

import networkx as nx
import numpy as np

from graph import create_graph


class ACDFrame(ttk.Frame):
    layout_parameters = {'height': 500, 'width': 300,
                         'left': 0, 'top': 0, 'rigth': 0, 'bottom': 0,
                         'borderwidth': 0, 'relief': 'raised',
                         'pad_x': 10, 'pad_y': 5}
    acd_parameters = {}

    def __init__(self, app, threshold=0.05, min_candidate=0.15, max_candidate=1., resolution=0.001, start_candidate=0.5,
                 use_offsets=False, frame_length=2., overlap=2, **layout_parameters):
        super().__init__(master=app.parameters_frame)

        # ACD parameters
        self.threshold = threshold
        self.min_candidate = min_candidate
        self.max_candidate = max_candidate
        self.resolution = resolution
        self.start_candidate = start_candidate

        # Window parameters
        self.use_offsets = use_offsets
        self.frame_length = frame_length
        self.overlap = overlap

        # Layout
        self.height = layout_parameters['height']
        self.width = layout_parameters['width']
        self.padding = (layout_parameters['left'], layout_parameters['top'],
                        layout_parameters['rigth'], layout_parameters['bottom'])
        self.borderwidth = layout_parameters['borderwidth']
        self.relief = layout_parameters['relief']
        self.pad_x = layout_parameters['pad_x']
        self.pad_y = layout_parameters['pad_y']

        self['height'] = self.height
        self['width'] = self.width
        self['padding'] = self.padding
        self['borderwidth'] = self.borderwidth
        self['relief'] = self.relief

        self.grid(row=0, column=0, padx=self.pad_x, pady=self.pad_y)


class PianoRollFrame(ttk.Frame):
    def __init__(self, app, height=250, width=450, left=0, top=0, right=0, bottom=0, borderwidth=5, relief='raised',
                 pad_x=10, pad_y=5, toolbar_on=False):
        super().__init__(master=app.plot_frame, height=height, width=width)

        # Size
        self.height = height
        self.width = width

        # Parameters
        self['padding'] = (left, top, right, bottom)
        self['borderwidth'] = borderwidth
        self['relief'] = relief

        # Layout
        self.grid(row=0, column=0, padx=pad_x, pady=pad_y)

        # # Figure
        # fig = Figure(figsize=(self.width / 100, self.height / 100), dpi=100)
        # t = np.arange(0, 3, .01)
        # fig.add_subplot(111).plot(t, 2 * np.sin(2 * np.pi * t))
        #
        # canvas = FigureCanvasTkAgg(fig, master=self)
        # canvas.draw()
        # canvas.get_tk_widget().pack(side=tk.TOP, anchor=tk.NW)
        #
        # if toolbar_on:
        #     toolbar = NavigationToolbar2Tk(canvas, self)
        #     toolbar.update()
        #     canvas.get_tk_widget().pack(side=tk.TOP)


class FullGraphFrame(ttk.Frame):
    def __init__(self, app, height=250, width=450, left=0, top=0, right=0, bottom=0, borderwidth=5, relief='raised',
                 pad_x=10, pad_y=5):
        super().__init__(master=app.plot_frame, height=height, width=width)

        # Size
        self.height = height
        self.width = width

        self['padding'] = (left, top, right, bottom)
        self['borderwidth'] = borderwidth
        self['relief'] = relief

        self.grid(row=1, column=0, padx=pad_x, pady=pad_y)


class GridFrame(ttk.Frame):
    def __init__(self, app, height=250, width=450, left=0, top=0, right=0, bottom=0, borderwidth=0, relief='raised',
                 pad_x=10, pad_y=5):
        super().__init__(master=app.plot_frame, height=height, width=width)

        # Size
        self.height = height
        self.width = width

        self['padding'] = (left, top, right, bottom)
        self['borderwidth'] = borderwidth
        self['relief'] = relief

        self.grid(row=0, column=1, padx=pad_x, pady=pad_y)

        self.grid_slider = None

        # Figure
        timestamps = np.array([0., 0.98, 1.52])
        self.plot_grid(timestamps, app.acd_frame)

    def plot_grid(self, timestamps, acd_frame: ACDFrame):
        threshold = acd_frame.threshold
        min_cand = acd_frame.min_candidate
        max_cand = acd_frame.max_candidate
        res_cand = acd_frame.resolution
        start_cand = acd_frame.start_candidate

        n = len(timestamps)
        notes = np.zeros(n)

        fig = Figure(figsize=(self.width / 100, self.height / 100), dpi=100)

        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, expand=True, fill=tk.BOTH, anchor=tk.NW)

        ax = fig.gca()
        points = ax.scatter(timestamps, notes)

        fig.subplots_adjust(bottom=0.35)

        ax.set_xlabel('Time (s)')
        ax.get_yaxis().set_ticks([])
        if not len(notes) == 0:
            ax.set_ylim([np.min(notes) - 2, np.max(notes) + 2])
        else:
            ax.set_ylim([-0.1, 0.5])

        line = ()
        for i in range(n):
            line = lines.Line2D([timestamps[i] - threshold, timestamps[i] + threshold], [notes[i], notes[i]], color='r')
            ax.add_line(line)

        def update(acd, idx):
            x_lim = ax.get_xlim()
            new_ticks = np.concatenate((np.arange(timestamps[idx], timestamps.min() - acd, -acd),
                                        np.arange(timestamps[idx], timestamps.max() + acd, acd)))
            ax.set_xticks(new_ticks, minor=False)
            ax.set_xlim(x_lim)
            ax.xaxis.grid(True, which='major')

            self.grid_slider.valtext.set_text(str(round(acd, 3)) + ' s')
            # note_slider.valtext.set_text(str(round(timestamps[idx], 3)) + ' s')

        ax.legend([points, line], ['timestamps', 'threshold'])

        ax_slider = fig.add_axes([0.15, 0.05, 0.65, 0.05])
        self.grid_slider = Slider(
            ax=ax_slider,
            label='aCD',
            valmin=min_cand,
            valmax=max_cand,
            valinit=start_cand,
            valstep=res_cand
        )

        self.grid_slider.on_changed(lambda val: update(val, 0))

        update(start_cand, 0)


class FocusedGraphFrame(ttk.Frame):
    frame_size = 3
    color = 0.9

    def __init__(self, app, height=250, width=450, left=0, top=0, right=0, bottom=0, borderwidth=5, relief='raised',
                 pad_x=10, pad_y=5):
        super().__init__(master=app.plot_frame, height=height, width=width)

        # Size
        self.height = height
        self.width = width

        self['padding'] = (left, top, right, bottom)
        self['borderwidth'] = borderwidth
        self['relief'] = relief

        self.grid(row=1, column=1, padx=pad_x, pady=pad_y)

        # Figure
        timestamps = np.array([0., 0.98, 1.52, 2.0, 2.53, 3.03])
        self.graph_dict = self.plot_graph(timestamps, app)

    def onclick(self, event, app):
        nodes = self.graph_dict['nodes']
        nodes.remove()

        canvas = event.canvas
        fig = canvas.figure
        ax = fig.gca()

        graph = self.graph_dict['graph']
        pos = nx.get_node_attributes(graph, 'pos')
        labels = self.graph_dict['labels']

        color_circles = self.color * np.ones((len(graph), 3))
        nodelist = list(graph)
        for n, node in enumerate(nodelist):
            node_x = pos[node][0]
            node_y = pos[node][1]
            distance = np.sqrt((node_x - event.xdata)**2 + (node_y - event.ydata)**2)
            if distance < 0.5:
                # print(node)
                color_circles[n, :] = np.array([1., 0.8, 0.8])
                app.grid_frame.grid_slider.set_val(float(labels[node].get_text()))
                break

        self.graph_dict['nodes'] = nx.draw_networkx_nodes(graph, pos=pos, ax=ax, node_size=800,
                                                          node_color=color_circles, node_shape='o')

        canvas.draw()

    def plot_graph(self, timestamps, app, current_frame=1):
        acd_frame = app.acd_frame
        threshold = acd_frame.threshold
        min_cand = acd_frame.min_candidate
        max_cand = acd_frame.max_candidate
        res_cand = acd_frame.resolution

        graph = create_graph(timestamps, threshold=threshold, min_cand=min_cand, max_cand=max_cand, res_cand=res_cand,
                             frame_size=self.frame_size, error_weight=1., tempo_var_weight=1.)

        pos = nx.get_node_attributes(graph, 'pos')
        node_labels = {idx: round(graph.nodes[idx]['acd'], 3) for idx in graph.nodes}

        color_circles = self.color * np.ones((len(graph), 3))

        fig = Figure(figsize=(self.width / 100, self.height / 100), dpi=100)
        fig.subplots_adjust(bottom=0., right=1., left=0., top=1.)

        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, expand=True, fill=tk.BOTH, anchor=tk.NW)

        ax = fig.gca()
        ax.axis('off')
        center_frame = current_frame - 1
        ax.set_xlim([center_frame - 1 - 0.2, center_frame + 1 + 0.2])

        # Plot graph
        nodes = nx.draw_networkx_nodes(graph, pos=pos, ax=ax, node_size=800, node_color=color_circles, node_shape='o')
        edges = nx.draw_networkx_edges(graph, pos=pos, ax=ax, arrows=True)
        labels = nx.draw_networkx_labels(graph, pos=pos, ax=ax, labels=node_labels, font_size=12)

        # Update vertical limits
        y_lim = ax.get_ylim()
        ax.set_ylim([y_lim[0] - 0.5, y_lim[1] + 0.2])

        # Draw
        canvas.draw()

        fig.canvas.mpl_connect('button_press_event', lambda event: self.onclick(event, app))

        return {'graph': graph, 'nodes': nodes, 'edges': edges, 'labels': labels}


class RhythmTranscriptionApp(tk.Tk):
    geometry_parameters = {'width': 1300, 'height': 600, 'x_offset': 100, 'y_offset': 50}

    def __init__(self, title='Rhythm Transcription'):
        super().__init__()

        # Window size and positioning
        # geometry_string = str(width) + 'x' + str(height) + '+' + str(x_offset) + '+' + str(y_offset)
        # self.geometry(geometry_string)

        # Title
        self.title(title)

        # Parameters frame
        self.parameters_frame = ttk.Frame(master=self)
        self.parameters_frame.grid(row=0, column=1)

        self.acd_frame = ACDFrame(app=self, **ACDFrame.layout_parameters)

        # Plot frame
        self.plot_frame = ttk.Frame(master=self)
        self.plot_frame.grid(row=0, column=0)

        self.piano_roll_frame = PianoRollFrame(self)
        self.full_graph_frame = FullGraphFrame(self)
        self.grid_frame = GridFrame(self)
        self.focused_graph_frame = FocusedGraphFrame(self)


if __name__ == "__main__":
    rhythm_app = RhythmTranscriptionApp()
    rhythm_app.mainloop()
