import os, errno
from PIL import Image
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib.ticker import LogLocator
from matplotlib.colors import LogNorm
from matplotlib import cm

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def list_dir(path):
    filter_dir = lambda x: os.path.isdir(os.path.join(path,x))
    filter_file = lambda x: os.path.isfile(os.path.join(path,x)) and not x.startswith('.') \
    and not x.split('.')[-1] in ['pyc', 'py','txt']
    ret = [n for n in os.listdir(path) if filter_dir(n) or filter_file(n)]
    return ret

def load_image(path):
    return Image.open(path)

def load_audio(path):
    return sf.read(path)

def plot_heatmap(arr, fname, pred=''):
    arr = np.flip(arr.mean(0), axis=0)
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)
    ax.imshow(arr[::-1], cmap='magma', interpolation='nearest')
    plt.ylim(plt.ylim()[::-1])
    plt.tight_layout()
    ax.text(.99, .98, pred, fontsize=20,
        color='white',
        fontweight='bold',
        verticalalignment='top',
        horizontalalignment='right',
        transform=ax.transAxes)
    plt.savefig(fname, format='png')
    plt.close()

class Val:
    sampling_rate = 22050
    duration = 4
    padmode = 'constant'
    samples = int(sampling_rate * duration)

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
