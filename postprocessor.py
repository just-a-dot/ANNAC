import wave
import numpy as np

def numpyToWav(data, outname):
    '''
    A function to reverse the preprocessing step and outputting a wave file with the given data.

    :param data: the audio data to be put in the file
    :param outname: The name of the file we want to create

    :returns: the outname of the file again.
    '''
    with wave.open(outname, 'wb') as f:
        channels = 1
        samplewidth = 2
        framerate = 22050
        f.setparams((channels, samplewidth, framerate, 0, 'NONE', 'not compressed'))
        for sample in data:
            for frame in sample.flatten().tolist():
                # Reverse the transformation from the preprocessor
                frame = frame * 2
                frame = frame - 1
                frame = round(frame * 32767) 
                f.writeframes((frame.to_bytes(2, byteorder='little', signed=True)))
    return outname