from utils import plot_heatmap
from model import MelspectrogramStretch

import torch
import numpy as np
import matplotlib
from matplotlib import pyplot
matplotlib.use('Agg')

class AudioInference:
    def __init__(self, model, transforms):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.transforms = transforms
        self.mel = MelspectrogramStretch(norm='db')
        self.mel.eval()

    def infer(self, data):
        sig_t, sr, _ = self.transforms.apply(data, None)
        length = torch.tensor(sig_t.size(0))
        sr = torch.tensor(sr)
        data = [d.unsqueeze(0).to(self.device) for d in [sig_t, length, sr]]
        label, conf = self.model.predict( data )
        return label, conf

    def draw(self, sig, sr, label, name):
        sig = np.float64(np.stack((sig, sig)).transpose(1, 0))
        sig = torch.tensor(sig).mean(dim=1).view(1,1,-1).float()
        spec = self.mel(sig)[0]
        out_path = f"real_time_data/{name}_pred.png"
        pred_txt = f"{label}"
        plot_heatmap(spec.cpu().numpy(), out_path, pred=pred_txt)