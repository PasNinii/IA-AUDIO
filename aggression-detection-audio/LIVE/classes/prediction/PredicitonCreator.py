from classes.prediction.TorchaudioPrediction import TorchaudioPrediction

class PredictionCreator:
    def create(self, network, device, transform):
        return TorchaudioPrediction(network, device, transform)

    def __call__(self, network, device, transform):
        return self.create(network, device, transform)