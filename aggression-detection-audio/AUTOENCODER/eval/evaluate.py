import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import MelspectrogramStretch
from utils import plot_heatmap, mkdir_p

from collections import defaultdict

class ClassificationEvaluator(object):
    def __init__(self, data_loader, model):
        self.data_loader = data_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.mel = MelspectrogramStretch(norm='db').to(self.device)
        self.criterion = F.mse_loss

    def evaluate(self, debug=False):
        with torch.no_grad():
            losses = []
            result = defaultdict(list)
            for batch_idx, batch in enumerate(tqdm(self.data_loader)):
                batch = [b.to(self.device) for b in batch]
                data, target_label, target_class = batch[:-2], batch[-2], batch[-1]
                output, target = self.model(data)
                loss = self.criterion(target, output)
                result[batch_idx] = {
                    "loss": loss.item(),
                    "target_label": target_label.item(),
                    "target_class": target_class.item(),
                }
            return result