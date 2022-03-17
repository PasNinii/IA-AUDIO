import os
import torch

import pandas as pd
import torch.utils.data as data

from data.Datasets import Dataset
from torch.utils.data.dataloader import default_collate
from utils.AudioRead import AudioRead


class SoundManager(object):
    def __init__(self, conf):
        audioRead = AudioRead()
        # Private
        self._path = conf["path"]
        self._args = conf["loader"]
        self._loading_function = audioRead.sound_file
        self._df = pd.read_csv(os.path.join(self._path, 'metadata/audio_test.csv'))
        self._df = self._remove_to_small()
        self._dataset = self._train_test_split()

    def _remove_to_small(self, threshold: float = 1) -> pd.DataFrame:
        """Remove audio file that are below a duration threshold

        Args:
            df (pd.DataFrame): DataFrame containing informations about urbansounds file
            min_sec (float, optional): threshold. Defaults to 0.5.

        Returns:
            df (pd.DataFrame): DataFrame with remaining values that are above the duration threshold
        """
        condition = (self._df['end'] - self._df['start']) >= threshold
        return self._df[(condition)]

    def _train_test_split(self):
        """Return a tuple dict train & validation data

        Returns:
            ret (dict): dict of dict containing train & validation data
                * ret.keys() = ["train", "val"]
                * value example
                    * {'path': 'path/UrbanSound8K/audio/fold9/101729-0-0-21.wav',
                       'class': 'air_conditioner',
                       'label': 0}
        """
        dataset = {}
        for type_ in ["train", "valid", "test"]:
            df = self._df[self._df['fold'] == type_]
            dataset[type_] = []
            for row in df[["slice_file_name", "classID", "label", "fold"]].values:
                fname = os.path.join(self._path, f"audio/{row[-1]}", row[0])
                dataset[type_].append({
                    "path": fname,
                    "classID": row[1],
                    "label": row[2]
                })
        return dataset

    def _pad_seq(self, batch):
        sort_ind = 0
        sorted_batch = sorted(batch, key=lambda x: x[0].size(sort_ind), reverse=True)
        seqs, srs, labels, classId = zip(*sorted_batch)
        srs, labels, classId = map(torch.LongTensor, [srs, labels, classId])
        seqs_pad = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)
        return seqs_pad, srs, labels, classId

    def get_training_loader(self, name, transform):
        """Return dataloader containing data

        Args:
            name (str): train, validation
            transfs (torch.data.utils.transforms): pytorch transformation

        Returns:
            data (torch.data.utils.dataloader): training or validation data
        """
        dataset = Dataset(self._dataset[name], loading_function=self._loading_function, transform=transform)
        return data.DataLoader(dataset=dataset, **self._args, collate_fn=self._pad_seq)

    def get_testing_loader(self, transform):
        """Return dataloader containing data

        Args:
            name (str): train, validation
            transfs (torch.data.utils.transforms): pytorch transformation

        Returns:
            data (torch.data.utils.dataloader): training and validation data
        """
        self._args["batch_size"] = 1
        dataset = Dataset(self._dataset["test"], loading_function=self._loading_function, transform=transform)
        return data.DataLoader(dataset=dataset, **self._args, collate_fn=self._pad_seq)


