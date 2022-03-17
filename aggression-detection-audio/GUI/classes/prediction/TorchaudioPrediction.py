from classes.prediction.AbstractPrediction import AbstractPrediction
from classes.Inference import AudioInference

class TorchaudioPrediction(AbstractPrediction):
    def __init__(self, network, device, transform):
        super().__init__(network, device, transform)
        self.audioInference = AudioInference(network, self.transform)

    def transform(self):
        return None

    def predict(self):
        label, conf = self.audioInference.infer(self.data)
        return label

    def save_specgram(self, label, sr):
        self.audioInference.draw(self.data, sr, label, self.j)