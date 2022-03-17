import numpy as np
import torch
import torch.nn as nn

from torchaudio.transforms import Spectrogram, MelSpectrogram , ComplexNorm
from torchaudio.transforms import TimeStretch, AmplitudeToDB
from torch.distributions import Uniform

def _num_stft_bins(lengths, fft_length, hop_length, pad):
    return (lengths + 2 * pad - fft_length + hop_length) // hop_length

class AutoencoderMelSpec(MelSpectrogram):
    def __init__(self, hop_length=None,
                       sample_rate=22050,
                       num_mels=128,
                       fft_length=2048,
                       norm='whiten'):

        super(AutoencoderMelSpec, self).__init__(sample_rate=sample_rate,
                                                    n_fft=fft_length,
                                                    hop_length=hop_length,
                                                    n_mels=num_mels)

        self.stft = Spectrogram(n_fft=self.n_fft, win_length=self.win_length,
                                       hop_length=self.hop_length, pad=self.pad,
                                       power=None, normalized=False)

        self.complex_norm = ComplexNorm(power=2.)
        self.norm = SpecNormalization(norm)

    def forward(self, x):
        x = self.stft(x)

        x = self.complex_norm(x)
        x = self.mel_scale(x)

        # Normalize melspectrogram
        x = self.norm(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'




class SpecNormalization(nn.Module):
    def __init__(self, norm_type, top_db=80.0):
        super(SpecNormalization, self).__init__()
        if 'db' == norm_type:
            self._norm = AmplitudeToDB(stype='power', top_db=top_db)
        elif 'whiten' == norm_type:
            self._norm = lambda x: self.z_transform(x)
        else:
            self._norm = lambda x: x

    def z_transform(self, x):
        # Independent mean, std per batch
        mean = x.mean()
        std = x.std()
        x = (x - mean) / std
        return x

    def forward(self, x):
        return self._norm(x)
