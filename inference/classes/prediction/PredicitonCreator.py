from classes.prediction.AbstractPrediction import AbstractPrediction
from classes.prediction.ConvPrediction import ConvPrediction
from classes.prediction.AutoPrediciton import AutoPrediction

class PredictionCreator:
    def create(self, network, device, transform, type) -> None:
        if type == "auto":
            return AutoPrediction(network, device, transform)
        elif type == "conv":
            return ConvPrediction(network, device, transform)
        else:
            raise(f"Exception: type {type} is not a valid for class PredictionCreator")

    def __call__(self, network, device, transform) -> AbstractPrediction:
        return self.create(network, device, transform)