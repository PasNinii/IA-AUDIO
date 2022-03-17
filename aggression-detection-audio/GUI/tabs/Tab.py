from classes.AudioHandler import AudioHandler

import model as mdl
import abc, torch
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import data as dt


def _get_transform(config, name):
    tsf_mode = config["data"]["format"]
    tsf_name = config["transforms"]["type"]
    tsf_args = config["transforms"]["args"]
    return getattr(dt, tsf_name)(name, tsf_mode, tsf_args)

def _get_model_att(checkpoint):
    m_name = checkpoint["config"]["model"]["type"]
    sd = checkpoint["state_dict"]
    classes = checkpoint["classes"]
    return m_name, sd, classes

class Tab(tk.Frame):
    def __init__(self, parent=None, name: str = "WavTab", network_file: str="", device="cpu"):
        super().__init__(parent)
        self.parent = parent
        self.name = name
        self.v = "undefined"
        self.filename = ""
        self.device = device
        self.network = None
        self.transform = None
        self.init_model(network_file)
        self.audioHandler = AudioHandler(self.network, self.device, self.transform)
        self.id = None

    def init_widgets(self):
        self.tab = ttk.Frame(self.parent)
        self.parent.add(self.tab, text=self.name)
        self.parent.pack(expand=1, fill="both")
        self.label = tk.Label(self.tab)
        self.label.pack()
        self.label.configure(text=f"label: {self.v}")

    def init_model(self, network_file):
        print(network_file)
        checkpoint = torch.load(network_file, map_location=self.device)
        config = checkpoint["config"]
        m_name, sd, classes = _get_model_att(checkpoint)
        self.network = getattr(mdl, m_name)(classes, config, state_dict=sd)
        self.network.load_state_dict(checkpoint["state_dict"])
        self.transform = _get_transform(config, "val")

    def play(self):
        self.update_label()

    def stop(self):
        try:
            self.audioHandler.stop()
            self.label.after_cancel(self.id)
        except ValueError:
            print(ValueError)

    @abc.abstractmethod
    def update_label(self):
        pass