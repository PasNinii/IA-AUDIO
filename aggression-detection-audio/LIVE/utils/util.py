import librosa, time
import numpy as np

from classes.decorators import timeit
from data.transforms import AudioTransforms


class Val:
    sampling_rate = 22050
    duration = 4
    padmode = 'constant'
    samples = int(sampling_rate * duration)

@timeit
def read_audio(path: str = "") -> np.ndarray:
    """Read audio file and transforms it into numpy.ndarray

    Args:
        path (str, optional): path where the audio file is located. Defaults to "".

    Returns:
        np.ndarray: Return numpy.ndarray containing the audio waves
    """
    try:
        sound, sample_rate = librosa.load(path,
                                          sr=Val.sampling_rate)
        if (len(sound) > Val.samples):
            sound = sound[0:0 + Val.samples]
        else:
            padding = Val.samples - len(sound)
            offset = padding // 2
            sound = np.pad(sound,
                           (offset, Val.samples - len(sound) - offset),
                           Val.padmode)
    except ValueError:
        print(ValueError)
    return sound, sample_rate

@timeit
def load_audio(path: str = "") -> (np.ndarray, int):
    """Read audio without tranformation

    Args:
        path (str, optional): Path to the audio file. Defaults to ""

    Returns:
        tuple: (sound wave, sample rate)
    """
    sound, sr = librosa.load(path, sr=Val.sampling_rate)
    return sound, sr


def _get_transform(config, name):
    tsf_args = config["transforms"]["args"]
    return AudioTransforms(tsf_args)


def _get_model_att(checkpoint):
    m_name = checkpoint["config"]["model"]["type"]
    sd = checkpoint["state_dict"]
    classes = checkpoint["classes"]
    return m_name, sd, classes
