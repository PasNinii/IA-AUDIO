# * Data Modules
import numpy as np
import pandas as pd
# * Sound Modules
from pydub import AudioSegment
# * Utils Modules
from tqdm import tqdm
import random
import os

def create_noise_sound(df: pd.DataFrame, save_new_audio_path: str, new_df_name: str, save_new_df_path: str) -> None:
    """Add environment noise to the actor dataset

    Arguments:
        df {pd.DataFrame} -- dataframe containing actors reference to audio files
        save_new_audio_path {str} -- path where to save newly created audio
        new_df_name {str} -- path where to save newly created audio
        save_new_df_path {str} -- path where to save dataframe containing reference to newly created audio files
    """
    global NEW_DF, COUNTER
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            mix_noise_sound(i=i, row=row, save_new_audio_path=save_new_audio_path)
            if (i % 100 == 0):
                NEW_DF.to_csv(f"{save_new_df_path}{new_df_name}.csv", index=False)
        except AttributeError:
            print(FileNotFoundError)
    NEW_DF.to_csv(f"{save_new_df_path}{new_df_name}_final", index=False)


def mix_noise_sound(i: int, row, save_new_audio_path: str) -> None:
    global COUNTER, NEW_DF
    random_index = []
    random_index = generate_random_index(6, len(NOISES))
    for index in random_index:
        sound_1 = AudioSegment.from_file(f"../resources/UrbanSound8K/audio/fold{row.fold}/{row.slice_file_name}")
        sound_2 = AudioSegment.from_file(f"../resources/noise_env/{NOISES[index]}")
        combined = sound_1.overlay(sound_2)
        slice_file_name = f"{str(COUNTER)}.wav"
        combined.export(f"{save_new_audio_path}fold{row.fold}/{slice_file_name}", format="wav")
        COUNTER += 1
        temp_df = pd.DataFrame(
            {
                "slice_file_name": [slice_file_name],
                "fsID": [row.fsID],
                "start": [row.start],
                "end": [row.end],
                "salience": [row.salience],
                "fold": [row.fold],
                "classID": [row.classID],
                "class": [row["class"]],
                "origin_file": [row.slice_file_name]
            }
        )
        NEW_DF = NEW_DF.append(temp_df, ignore_index=True)


def generate_random_index(n: int, l: int) -> list:
    """Generate list of random numbers

    Arguments:
        n {int} -- number of indexes to generate
        l {int} -- size of the dataframe containing noise references

    Returns:
        list -- list of random numbers
    """
    random_index = []
    for i in range(n):
        random_index.append(random.randint(0, l-1))
    return random_index

if __name__ == "__main__":
    COUNTER = 49476
    NOISES = os.listdir("../resources/noise_env/")
    NEW_DF = pd.DataFrame(columns=[
                    "slice_file_name",
                    "fsID",
                    "start",
                    "end",
                    "salience",
                    "fold",
                    "classID",
                    "class"
    ])
    df = pd.read_csv("../resources/UrbanSound8K/metadata/UrbanSound8K.csv")
    df = df.iloc[8340:len(df)]
    create_noise_sound(df,
                       "../test/audio/",
                       "new_df",
                       "../test/csv/")