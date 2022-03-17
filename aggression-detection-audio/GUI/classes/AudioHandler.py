from classes.prediction.PredicitonCreator import PredictionCreator

import pyaudio

class AudioHandler(object):
    def __init__(self, network, device, transform):
        predictionCreator = PredictionCreator()
        self.prediction = predictionCreator.create(network, device, transform)
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 1024
        self.DURATION = 2
        self.result = None
        self.p = None
        self.stream = None
        self.i = 0

    def start(self):
        print("Mic open")
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  output=False,
                                  stream_callback=self.callback,
                                  frames_per_buffer=self.CHUNK)

    def stop(self):
        try:
            self.stream.close()
            self.p.terminate()
            print("Audio stream terminated")
        except AttributeError:
            print(f"{AttributeError}: make sure you started the stream in the first place")

    def callback(self, in_data, frame_count, time_info, flag):
        self.i += 1
        if (self.i < self.RATE / self.CHUNK * self.DURATION):
            self.prediction.read(in_data)
        else:
            self.predict()
        return None, pyaudio.paContinue

    def predict(self):
        self.prediction.prepare()
        self.result = self.prediction.predict()
        self.prediction.save_specgram(self.result, self.RATE)
        self.prediction.reset_data()
        self.i = -1
