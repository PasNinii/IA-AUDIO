import os, errno
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def list_dir(path):
    filter_dir = lambda x: os.path.isdir(os.path.join(path,x))
    filter_file = lambda x: os.path.isfile(os.path.join(path,x)) and not x.startswith('.') \
    and not x.split('.')[-1] in ['pyc', 'py','txt']
    ret = [n for n in os.listdir(path) if filter_dir(n) or filter_file(n)]
    return ret

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