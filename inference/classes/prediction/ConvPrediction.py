from classes.prediction.AbstractPrediction import AbstractPrediction
from classes.inference.ConvInference import ConvInference

class ConvPrediction(AbstractPrediction):
    def __init__(self, network, device, transform) -> None:
        super().__init__(network, device, transform)
        self.inference = ConvInference(network, device, self.transform)

    def predict(self, rate: int = 44100, threshold: float = 0.5, model_type: str = "crnn") -> str:
        super().predict(rate=rate, threshold=threshold, model_type=model_type)