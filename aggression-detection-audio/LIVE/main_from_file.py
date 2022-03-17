from classes.decorators import timeit, d
from classes.Inference import AudioInference
from data.transforms import AudioTransforms
import model as mdl
from utils.util import read_audio, load_audio, _get_model_att, _get_transform

import os, time, torch, warnings
from collections import defaultdict
import pandas as pd

warnings.filterwarnings("ignore")


@timeit
def prediction(i):
    sound, sr = load_audio(f"./audio/output_{i}.mp3")
    label, pred = audioInference.infer(sound)
    i += 1
    return i

def main():
    global audioInference
    # ConvNet_0715_105845
    network_file = os.path.join(os.path.abspath("."), "saved_cv", "AudioCRNN_0710_145816", "checkpoints", "model_best.pth")
    device = "cpu"
    checkpoint = torch.load(network_file, map_location=device)
    config = checkpoint["config"]
    m_name, sd, classes = _get_model_att(checkpoint)
    network = getattr(mdl, m_name)(classes, config, state_dict=sd)
    network.load_state_dict(checkpoint["state_dict"])
    transform = _get_transform(config, "val")
    audioInference = AudioInference(network, transform)
    i = 0
    j = 0
    redo = True
    while(j < 104):
        i = prediction(i)
        j += 1
        i = j % 7
        time.sleep(3)



if __name__ == "__main__":
    main()
    pd.DataFrame.from_dict(d, orient='columns').to_csv("results/result.csv", index=False)
