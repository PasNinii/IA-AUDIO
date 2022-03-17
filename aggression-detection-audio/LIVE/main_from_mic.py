import pyaudio
import socket
from threading import Thread


from classes.AudioHandler import AudioHandler
from classes.decorators import timeit, d
from classes.Inference import AudioInference
from data.transforms import AudioTransforms
import model as mdl
from utils.util import _get_model_att, _get_transform

import os, time, torch, warnings
from collections import defaultdict


network_file = os.path.join(os.path.abspath("."), "saved_cv", "ConvNet_0715_105845", "checkpoints", "model_best.pth")
device = "cpu"
checkpoint = torch.load(network_file, map_location=device)
config = checkpoint["config"]
m_name, sd, classes = _get_model_att(checkpoint)
network = getattr(mdl, m_name)(classes, config, state_dict=sd)
network.load_state_dict(checkpoint["state_dict"])
transform = _get_transform(config, "val")

if __name__ == "__main__":
    audioHandler = AudioHandler(network, device, transform)
    Ts = Thread(target=audioHandler.server)
    Tp = Thread(target=audioHandler.read)
    Ts.setDaemon(True)
    Tp.setDaemon(True)
    Ts.start()
    Tp.start()
    Ts.join()
    Tp.join()