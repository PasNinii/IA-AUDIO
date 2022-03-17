from classes.inference.AbstractInference import AbstractInference
from transforms import AutoencoderMelSpec

import torch.nn.functional as F

class AutoInference(AbstractInference):
    def __init__(self, model, device, transforms) -> None:
        super().__init__(model, device, transforms)
        self.mel = AutoencoderMelSpec(norm='whiten')
        self.mel.eval()
        self._criterion = F.mse_loss

    def infer(self, data) -> tuple:
        data, _ = self.transforms.apply(data)
        data = data.unsqueeze(0).to(self.device)
        origin, target = self.model(data)
        threshold = self._criterion(origin, target)
        return "Unknown", threshold.item()

