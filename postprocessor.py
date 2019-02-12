import wave
import numpy as np

def numpyToWav(data, outname):
    with wave.open(outname, 'wb') as f:
        channels = 1
        samplewidth = 2
        framerate = 22050
        f.setparams((channels, samplewidth, framerate, 0, 'NONE', 'not compressed'))
        for sample in data:
            for frame in sample.flatten().tolist():
                frame = round(frame * 32767) # Reverse the transformation from the preprocessor
                f.writeframes((frame.to_bytes(2, byteorder='little', signed=True)))


