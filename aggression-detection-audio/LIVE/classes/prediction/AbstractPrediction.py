import abc
import numpy as np

class AbstractPrediction(metaclass=abc.ABCMeta):
    def __init__(self, network, device, transform):
        self.network = network
        self.device = device
        self.transform = transform
        self.data = []
        self.tensor = []
        self.j = 0

    def read(self, in_data):
        data = np.fromstring(in_data, dtype=np.float32)
        data = np.nan_to_num(data)
        self.data.append(data)

    def prepare(self):
        self.tensor = np.concatenate(self.data).ravel()

    def reset_data(self):
        self.data = []
        self.tensor = []
        self.j += 1

    @abc.abstractmethod
    def predict(self):
        pass