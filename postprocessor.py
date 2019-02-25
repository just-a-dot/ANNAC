import wave
import numpy as np

def numpyToWav(data, outname):
    with wave.open(outname, 'wb') as f:
        channels = 1
        samplewidth = 2
        framerate = 22050
        f.setparams((channels, samplewidth, framerate, 0, 'NONE', 'not compressed'))
        for sample in data:
            for frame in sample:
                frame = frame * 2
                frame = frame - 1
                frame = round(frame * 32767) # Reverse the transformation from the preprocessor
                f.writeframes((frame.to_bytes(2, byteorder='little', signed=True)))

'''
File "main.py", line 105, in <module>
    postprocessor.numpyToWav(base, names[i] + '-out.wav')
  File "/data/postprocessor.py", line 14, in numpyToWav
    frame = round(frame * 32767) # Reverse the transformation from the preprocessor
TypeError: type numpy.ndarray doesn't define __round__ method
'''