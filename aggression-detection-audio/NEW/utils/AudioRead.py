import soundfile as sf


class AudioRead(object):
    def __init__(self):
        super().__init__()

    def sound_file(self, path):
        return sf.read(path)