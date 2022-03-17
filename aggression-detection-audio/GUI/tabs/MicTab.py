from tabs.Tab import Tab

import tkinter as tk

class MicTab(Tab):
    def __init__(self, parent=None, name: str = "WavTab", network_file: str = "", device="cpu"):
        self.mic_state = "close"
        super().__init__(parent, name, network_file, device)
        self.init_widgets()

    def init_widgets(self):
        super().init_widgets()
        self.label_mic = tk.Label(self.tab)
        self.label_mic.pack()
        self.label_mic.configure(text=f"mic: {self.mic_state}")
        self.playButton = tk.Button(self.tab, text="play", command=lambda: self.play())
        self.stopButton = tk.Button(self.tab, text="stop", command=lambda: self.stop())
        self.playButton.pack()
        self.stopButton.pack()

    def play(self):
        self.audioHandler.start()
        super().play()
        self.mic_state = "open"
        self.label_mic.configure(text=f"mic: {self.mic_state}")

    def stop(self):
        super().stop()
        self.mic_state = "close"
        self.label_mic.configure(text=f"mic: {self.mic_state}")

    def update_label(self):
        super().update_label()
        self.v = self.audioHandler.result
        self.label.configure(text=f"label: {self.v}")
        self.id = self.label.after(10, self.update_label)
