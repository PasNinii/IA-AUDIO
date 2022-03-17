import abc, torch
import tkinter as tk
from tkinter import ttk


class Tab(tk.Frame):
    def __init__(self, parent=None, name: str = "WavTab", device="cpu") -> None:
        super().__init__(parent)
        self.parent = parent
        self.name = name
        self.v = "undefined"
        self.filename = ""
        self.device = device
        self.network = None
        self.transform = None
        self.id = None

    def init_widgets(self) -> None:
        self.tab = ttk.Frame(self.parent)
        self.parent.add(self.tab, text=self.name)
        self.parent.pack(expand=1, fill="both")
        self.label = tk.Label(self.tab)
        self.label.pack()
        self.label.configure(text=f"label: {self.v}")
        self.label_mic = tk.Label(self.tab)
        self.label_mic.pack()
        self.label_mic.configure(text=f"mic: {self.mic_state}")
        self.playButton = tk.Button(self.tab, text="play", command=lambda: self.play())
        self.stopButton = tk.Button(self.tab, text="stop", command=lambda: self.stop())
        self.playButton.pack()
        self.stopButton.pack()

    @abc.abstractmethod
    def init_model(self, network_file) -> None:
        pass

    def play(self) -> None:
        self.audioHandler.start()
        self.update_label()
        self.mic_state = "open"
        self.label_mic.configure(text=f"mic: {self.mic_state}")

    def stop(self) -> None:
        try:
            self.audioHandler.stop()
            self.label.after_cancel(self.id)
            self.mic_state = "close"
            self.label_mic.configure(text=f"mic: {self.mic_state}")
        except ValueError:
            print(ValueError)

    def update_label(self) -> None:
        self.v = self.audioHandler.result
        self.label.configure(text=f"label: {self.v}")
        self.id = self.label.after(10, self.update_label)