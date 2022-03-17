import torch
import torch.nn as nn
import torch.nn.functional as F
from .BaseModel import BaseModel

from transforms.MelSpectrogram import MelspectrogramStretch


class AutoEncoderConv(nn.Module):
    def __init__(self):
        super(AutoEncoderConv,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=8,
                      kernel_size=3,
                      stride=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=8,
                      out_channels=16,
                      kernel_size=3,
                      stride=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=5,
                      stride=2),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32,
                               out_channels=16,
                               kernel_size=5,
                               output_padding=1,
                               stride=2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=16,
                               out_channels=8,
                               kernel_size=3,
                               stride=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=8,
                               out_channels=1,
                               kernel_size=3,
                               stride=1),
            nn.ReLU(inplace=True),

        )

    def forward(self, x):
        x = x[:, :, :, 0:x.shape[3] - 1] if x.shape[3] % 2 != 0 else x
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class ConvAutoencoderNet(BaseModel):
    def __init__(self, config={}, state_dict=None):
        super(ConvAutoencoderNet, self).__init__()
        self.spec = MelspectrogramStretch(hop_length=None,
            num_mels=128,
            fft_length=2048,
            norm='whiten')

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=8,
                      kernel_size=3,
                      stride=1),
            nn.SELU(inplace=True),

            nn.Conv2d(in_channels=8,
                      out_channels=16,
                      kernel_size=3,
                      stride=1),
            nn.SELU(inplace=True),

            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=5,
                      stride=2),
            nn.SELU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32,
                               out_channels=16,
                               kernel_size=5,
                               output_padding=1,
                               stride=2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=16,
                               out_channels=8,
                               kernel_size=3,
                               stride=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=8,
                               out_channels=1,
                               kernel_size=3,
                               stride=1),
            nn.ReLU(inplace=True),

        )

    def forward(self, batch):
        x, _ = batch
        xt = x.float().transpose(1,2)
        x = self.spec(xt)
        x = x[:, :, :, 0:x.shape[3] - 1] if x.shape[3] % 2 != 0 else x
        y = x
        x = self.encoder(x)
        x = self.decoder(x)
        return x, y
