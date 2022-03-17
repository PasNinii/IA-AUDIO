import pyaudio
import socket
from threading import Thread

frames = []

def udpStream():
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    i = 0
    j = 0
    while True:
        if len(frames) > 0:
            udp.sendto(frames.pop(0), ("192.168.1.16", 12345))
        if i % 10000000 == 0:
            j += 1
            print(j)
        i += 1
    udp.close()

def record(stream, CHUNK):
    while True:
        frames.append(stream.read(CHUNK))

if __name__ == "__main__":
    CHUNK = 1024 * 2
    FORMAT = pyaudio.paFloat32
    CHANNELS = 2
    RATE = 44100

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    output=False,
                    frames_per_buffer=CHUNK,
                    )

    Tr = Thread(target = record, args = (stream, CHUNK,))
    Ts = Thread(target = udpStream)
    Tr.setDaemon(True)
    Ts.setDaemon(True)
    Tr.start()
    Ts.start()
    Tr.join()
    Ts.join()