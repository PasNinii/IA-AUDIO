import abc
import numpy as np

import scipy
import scipy.io
import scipy.io.wavfile

from datetime import datetime

from entity.AlarmEntity import AlarmEntity
from repository.AlarmRepository import AlarmRepository

import time

class AbstractPrediction(metaclass=abc.ABCMeta):
    def __init__(self, network, device, transform) -> None:
        self.repository = AlarmRepository()
        self.network = network
        self.device = device
        self.transform = transform
        self.data = []

    def read(self, in_data) -> None:
        data = np.fromstring(in_data, dtype=np.float32)
        data = np.nan_to_num(data)
        self.data.append(data)

    def save_audio(self, audio: np.array, rate: int = 44100, fname: str = datetime.now()) -> None:
        scipy.io.wavfile.write(f"../front/src/assets/audio/{fname}.wav", rate, audio)

    def prepare(self) -> None:
        self.data = np.concatenate(self.data).ravel()

    def reset_data(self) -> None:
        self.data = []

    def predict(self, rate: int = 44100, threshold: float = 0.5, model_type: str = "Unknown") -> str:
        audio = self.data
        label, conf = self.inference.infer(self.data)
        print(label, conf)
        if conf > threshold:
            fname = time.strftime('%Y%m%d-%H%M%S')
            entity = AlarmEntity(threshold=conf, audio_path=f"{fname}.wav", classe=label, model_type=model_type)
            self.repository.setEntity(entity)
            self.repository.insert()
            self.save_audio(audio, rate=rate, fname=fname)
        return label