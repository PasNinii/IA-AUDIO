import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

from .mel_spectrogram import MelspectrogramStretch


class AutoEncoderConv(nn.Module):
    def __init__(self):
        super(AutoEncoderConv,self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(6,16,kernel_size=5),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16,6,kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6,1,kernel_size=5),
            nn.ReLU(True))

        def forward(self,x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class ConvNet(BaseModel):
    def __init__(self, classes, config={}, state_dict=None):
        super(ConvNet, self).__init__()
        self.spec = MelspectrogramStretch(hop_length=None,
            num_mels=128,
            fft_length=2048,
            norm='whiten',
            stretch_param=[0.4, 0.4],
            random_crop=False)
        self.classes = classes

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(6,16,kernel_size=5),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16,6,kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6,1,kernel_size=5),
            nn.ReLU(True))


    def forward(self, batch):
        x, lengths, _ = batch
        xt = x.float().transpose(1,2)
        x, lengths = self.spec(xt, lengths)

        y = x

        x = self.encoder(x)
        x = self.decoder(x)

        return x, y
