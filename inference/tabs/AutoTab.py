from tabs.Tab import Tab

import torch

import model as mdl
import transforms as transforms

from classes.AudioHandler import AudioHandler


def _get_transform(config):
    tsf_name = config["transforms"]["type"]
    tsf_args = config["transforms"]["args"]
    return getattr(transforms, tsf_name)(tsf_args)

def _get_model_att(checkpoint) -> tuple:
    m_name = checkpoint["config"]["model"]["type"]
    sd = checkpoint["state_dict"]
    return m_name, sd

class AutoTab(Tab):
    def __init__(self, parent=None, name: str = "WavTab", network_file: str = "", device="cpu") -> None:
        self.mic_state = "close"
        super().__init__(parent, name, device)
        self.init_model(network_file)
        self.audioHandler = AudioHandler(self.network, self.device, self.transform, "auto")
        self.init_widgets()


    def init_model(self, network_file) -> None:
        checkpoint = torch.load(network_file, map_location=self.device)
        config = checkpoint["config"]
        m_name, sd = _get_model_att(checkpoint)
        self.network = getattr(mdl, m_name)(config, state_dict=sd)
        self.network.load_state_dict(checkpoint["state_dict"])
        self.transform = _get_transform(config)
