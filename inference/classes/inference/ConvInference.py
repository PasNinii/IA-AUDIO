from classes.inference.AbstractInference import AbstractInference
from utils import plot_heatmap
from transforms import ConvolutionalMelSpec

import torch
import numpy as np
import matplotlib
from matplotlib import pyplot
matplotlib.use('Agg')

class ConvInference(AbstractInference):
    def __init__(self, model, device, transforms) -> None:
        super().__init__(model, device, transforms)
        self.mel = ConvolutionalMelSpec(norm='whiten')
        self.mel.eval()

    def infer(self, data) -> tuple:
        sig_t, sr, _ = self.transforms.apply(data, None)
        length = torch.tensor(sig_t.size(0))
        sr = torch.tensor(sr)
        data = [d.unsqueeze(0).to(self.device) for d in [sig_t, length, sr]]
        label, conf = self.model.predict(data)
        return label, conf