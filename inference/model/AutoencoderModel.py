import torch
import torch.nn as nn
import torch.nn.functional as F
from .BaseModel import BaseModel

from transforms.AutoencoderMelSpec import AutoencoderMelSpec


class AutoEncoderConv(nn.Module):
    def __init__(self):
        super(AutoEncoderConv,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=3,
                      stride=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      stride=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=5,
                      stride=2),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64,
                               out_channels=64,
                               kernel_size=5,
                               output_padding=1,
                               stride=2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=64,
                               out_channels=32,
                               kernel_size=3,
                               stride=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=32,
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
        self.spec = AutoencoderMelSpec(hop_length=None,
            num_mels=64,
            fft_length=2048,
            norm='whiten')

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=3,
                      stride=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      stride=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=5,
                      stride=2),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64,
                               out_channels=64,
                               kernel_size=5,
                               output_padding=1,
                               stride=2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=64,
                               out_channels=32,
                               kernel_size=3,
                               stride=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=32,
                               out_channels=1,
                               kernel_size=3,
                               stride=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, batch):
        # batch (tensor, sampling_rate)
        x = batch
        # x.shape [batch size, tensor size, channels]
        xt = x.float().transpose(1,2)
        x = self.spec(xt)

        x = x[:, :, :, 0:x.shape[3] - 1] if x.shape[3] % 2 != 0 else x
        y = x
        x = self.encoder(x)
        x = self.decoder(x)
        return x, y
