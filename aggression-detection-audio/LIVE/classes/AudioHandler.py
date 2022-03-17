from classes.prediction.PredicitonCreator import PredictionCreator

import pyaudio, socket

class AudioHandler(object):
    def __init__(self, network, device, transform):
        _predictionCreator = PredictionCreator()
        self.result = None
        self._prediction = _predictionCreator.create(network, device, transform)
        self._FORMAT = pyaudio.paFloat32
        self._CHANNELS = 1
        self._RATE = 44100
        self._CHUNK = 1024 * 2
        self._DURATION = 2
        self._p = None
        self._stream = None
        self._i = 0
        self._PORT = 12345
        self._IPADDRESS = "192.168.1.10"

    def start(self):
        print("Mic open")
        self._p = pyaudio.PyAudio()
        self._stream = self.p.open(format=self._FORMAT,
                                   channels=self._CHANNELS,
                                   rate=self._RATE,
                                   output=True,
                                   frames_per_buffer=self._CHUNK)

    def server(self):
        upd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        upd.bind((self._IPADDRESS, self._PORT))
        print(f"Server")
        while(True):
            data, addr = upd.recvfrom(self._CHUNK * 2 * self._CHANNELS)
            self._prediction.read(data)
        udp.close()

    def read(self):
        print(f"Read")
        BUFFER = 100
        while(True):
            if(len(self._prediction.data) == BUFFER):
                self.predict()

    def predict(self):
        self._prediction.prepare()
        self.result = self._prediction.predict()
        print(self.result)
        self._prediction.reset_data()
        self._i = -1
