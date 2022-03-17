import model as mdl
import data as DTDHandler

import time
import tkinter as tk
from tkinter import ttk
from tabs.MicTab import MicTab

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.geometry('500x300')
        self.pack()
        self.parent = ttk.Notebook(self.master)
        self.value = 0
        self.count = 0
        self.id = 0
        self.init_widgets()

    def init_widgets(self):
        self.multiTab = MicTab(parent=self.parent,
                               name="CNN Classification",
                               network_file="./saved_cv/ConvNet_0715_105845/checkpoints/model_best.pth",
                               device="cpu")

        self.multiTab = MicTab(parent=self.parent,
                               name="CRNN Classification",
                               network_file="./saved_cv/AudioCRNN_0811_111731/checkpoints/model_best.pth",
                               device="cpu")
        self.label = tk.Label(self.master)
