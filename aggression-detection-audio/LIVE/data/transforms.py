import torch
import numpy as np
from torchvision import transforms

from classes.decorators import timeit

class AudioTransforms(object):
    def __init__(self, args):
        self.transfs = transforms.Compose([
                ProcessChannels(args['channels']),
                ToTensorAudio()
        ])

    @timeit
    def apply(self, data, target):
        audio, sr = data, 41000
        return self.transfs(audio), sr, target

    def __repr__(self):
        return self.transfs.__repr__()


class ProcessChannels(object):
    def __init__(self, mode):
        self.mode = mode


    def _modify_channels(self, audio, mode):
        if mode == 'mono':
            new_audio = audio if audio.ndim == 1 else audio[:,:1]
        elif mode == 'stereo':
            new_audio = np.stack([audio]*2).T if audio.ndim == 1 else audio
        elif mode == 'avg':
            new_audio= audio.mean(axis=1) if audio.ndim > 1 else audio
            new_audio = new_audio[:,None]
        else:
            new_audio = audio
        return new_audio


    def __call__(self, tensor):
        return self._modify_channels(tensor, self.mode)

    def __repr__(self):
        return self.__class__.__name__ + '(mode={})'.format(self.mode)


class ToTensorAudio(object):
    def __call__(self, tensor):
        return torch.from_numpy(tensor)

    def __repr__(self):
        return self.__class__.__name__ + '()'



