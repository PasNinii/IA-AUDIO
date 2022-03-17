from model import MelspectrogramStretch

from classes.decorators import timeit

import torch
import numpy as np

class AudioInference:
    def __init__(self, model, transforms):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.transforms = transforms
        self.mel = MelspectrogramStretch(norm='db')
        self.mel.eval()

    @timeit
    def infer(self, data):
        sig_t, sr, _ = self.transforms.apply(data, None)
        length = torch.tensor(sig_t.size(0))
        sr = torch.tensor(sr)
        data = [d.unsqueeze(0).to(self.device) for d in [sig_t, length, sr]]
        label, conf = self.model.predict( data )
        return label, conf