
import os
import pandas as pd
import numpy as np

import torch
import torch.utils.data as data
from torch.utils.data.dataloader import default_collate

from data.data_sets import Dataset
from utils.util import list_dir, load_image, load_audio, read_audio

class UrbanSoundManager(object):
    """
        * Class that enables the loading of the data
    """
    loading_formats = {
        "image": load_image,
        "audio_cnn": read_audio,
        "audio_crnn": load_audio
    }

    def __init__(self, config):
        self.path = config['path']
        self.args = config['loader']
        self.splits = config['splits']
        # temporaire activer qu'en mode test
        self.splits["val"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.loading_function = self.loading_formats[config['format']]
        urban_csv = os.path.join(self.path, 'metadata/UrbanSound8K.csv')
        urban_df = pd.read_csv(urban_csv).sample(frac=1)
        self.urban_df = self._remove_too_small(urban_df, 1)
        self.classes = self._get_classes(self.urban_df[['class', 'classID']])
        print('*' * 50)
        print(f"{len(self.classes)} classes:\n{self.classes}")
        print('*' * 50)
        self.data_splits = self._10kfold_split(self.urban_df)

    def _remove_too_small(self, df, min_sec=0.5):
        """Remove audio file that are below a duration threshold

        Args:
            df (pd.DataFrame): DataFrame containing informations about urbansounds file
            min_sec (float, optional): threshold. Defaults to 0.5.

        Returns:
            df (pd.DataFrame): DataFrame with remaining values that are above the duration threshold
        """
        dur_cond = (df['end'] - df['start'])>=min_sec
        not_train_cond = ~df['fold'].isin(self.splits['train'])
        return df[(dur_cond)|(not_train_cond)]

    def _get_classes(self, df):
        """Return classes name

        Args:
            df (pd.DataFrame): DataFrame containing informations about urbansounds file

        Returns:
            df (list): Return a list containing the classes label
        """
        c_col = df.columns[0]
        idx_col = df.columns[1]
        return df.drop_duplicates().sort_values(idx_col)[c_col].unique()

    def _10kfold_split(self, df):
        """Return a tuple containing train & validation data

        Args:
            df (pd.DataFrame): DataFrame containing informations about urbansounds file

        Returns:
            ret (dict): dict of dict containing train & validation data
                * ret.keys() = ["train", "val"]
                * value example
                    * {'path': 'path/UrbanSound8K/audio/fold9/101729-0-0-21.wav',
                       'class': 'air_conditioner',
                       'label': 0}
        """
        ret = {}
        for s, inds in self.splits.items():
            df_split = df[df['fold'].isin(inds)]
            ret[s] = []
            for row in df_split[['slice_file_name', 'classID', 'label', 'fold']].values:
                fold_mod = 'audio/fold%s'%row[-1]
                fname = os.path.join(self.path, fold_mod, '%s'%row[0])
                ret[s].append( {'path':fname, 'classID':row[1], 'label':row[2]} )
        print(f"Number of training data: {len(ret['train'])}\nNumber of validation data: {len(ret['val'])}")
        print('*' * 50)
        print('*' * 50)
        return ret

    def get_loader(self, name, transfs):
        """Return dataloader containing data

        Args:
            name (str): train, validation
            transfs (torch.data.utils.transforms): pytorch transformation

        Returns:
            data (torch.data.utils.dataloader): training, validation data
        """
        self.args["batch_size"] = 1
        dataset = Dataset(self.data_splits[name], loading_function=self.loading_function, transforms=transfs)
        return data.DataLoader(dataset=dataset, **self.args, collate_fn=self.pad_seq)


    def pad_seq(self, batch):
        sort_ind = 0
        sorted_batch = sorted(batch, key=lambda x: x[0].size(sort_ind), reverse=True)
        seqs, srs, labels, classId = zip(*sorted_batch)
        lengths, srs, labels, classId = map(torch.LongTensor, [[x.size(sort_ind) for x in seqs], srs, labels, classId])
        seqs_pad = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)
        return seqs_pad, lengths, srs, labels, classId


