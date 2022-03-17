import subprocess as sp
import time

ipaddress = "127.0.0.1"
port = 50000

index = 0
current_start = 0
current_end = 4
while (True):
    print(index)
    command = ["ffmpeg",
        '-protocol_whitelist', 'file,rtp,udp',
        '-i', 'audio.sdp',
        '-ss', f'{current_start}',
        '-t', f'{current_end}',
        f'audio/output_{index}.opus']
    print('commande:', command)
    sp.Popen(command)
    time.sleep(3)
    index += 1
