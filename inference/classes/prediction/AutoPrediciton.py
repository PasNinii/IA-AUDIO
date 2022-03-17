from classes.prediction.AbstractPrediction import AbstractPrediction
from classes.inference.AutoInference import AutoInference

class AutoPrediction(AbstractPrediction):
    def __init__(self, network, device, transform) -> None:
        super().__init__(network, device, transform)
        self.inference = AutoInference(network, device, self.transform)

    def predict(self, rate: int = 44100, threshold: float = 0.70, model_type: str = "autoencoder") -> str:
        super().predict(rate=rate, threshold=threshold, model_type=model_type)