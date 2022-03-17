
import numpy as np
import librosa
import matplotlib.pyplot as plt
import noisereduce as nr
import os
import pyaudio

CHUNKSIZE = 22050 # fixed chunk size
RATE = 22050

# initialize portaudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNKSIZE)

#noise window
data = stream.read(10000)
noise_sample = np.frombuffer(data, dtype=np.float32)
loud_threshold = np.mean(np.abs(noise_sample)) * 10
print("Loud threshold", loud_threshold)
audio_buffer = []
near = 0

while(True):
    # Read chunk and load it into numpy array.
    data = stream.read(CHUNKSIZE)
    current_window = np.frombuffer(data, dtype=np.float32)

    #Reduce noise real-time
    current_window = nr.reduce_noise(audio_clip=current_window, noise_clip=noise_sample, verbose=False)

    if(audio_buffer==[]):
        audio_buffer = current_window
    else:
        if(np.mean(np.abs(current_window))<loud_threshold):
            print("Inside silence reign")
            if(near<10):
                audio_buffer = np.concatenate((audio_buffer,current_window))
                near += 1
            else:
                predictSound(np.array(audio_buffer))
                audio_buffer = []
                near
        else:
            print("Inside loud reign")
            near = 0
            audio_buffer = np.concatenate((audio_buffer,current_window))

# close stream
stream.stop_stream()
stream.close()
p.terminate()