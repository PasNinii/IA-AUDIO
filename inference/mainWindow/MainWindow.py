import model as mdl

import time
import tkinter as tk
import tabs as tab
from tkinter import ttk

class Application(tk.Frame):
    def __init__(self, master=None) -> None:
        super().__init__(master)
        self.master = master
        self.master.geometry('500x300')
        self.pack()
        self.parent = ttk.Notebook(self.master)
        self.value = 0
        self.count = 0
        self.id = 0
        self.init_widgets()

    def init_widgets(self) -> None:
        self.multiTab = tab.AutoTab(parent=self.parent,
                               name="Autoencoder",
                               network_file="./saved_cv/ConvAutoencoderNet_0923_131616/checkpoints/model_best.pth",
                               device="cpu")

        self.multiTab = tab.ConvTab(parent=self.parent,
                               name="CRNN Classification",
                               network_file="./saved_cv/AudioCRNN_0923_113535/checkpoints/model_best.pth",
                               device="cpu")
        self.label = tk.Label(self.master)
