import wave
import numpy as np

def numpyToWav(data, outname):
    with wave.open(outname, 'wb') as f:
        channels = 1
        samplewidth = 2
        framerate = 44100
        f.setparams((channels, samplewidth, framerate, 0, 'NONE', 'not compressed'))
        for frame in data.tolist():
            f.writeframes((frame.to_bytes(2, byteorder='little', signed=True)))


