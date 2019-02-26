import wave
import subprocess
import numpy as np
import os

def fileToNpArray(f, size):
    res = []
    frames = []
    frame = f.readframes(1)
    while frame != b'': # b'' = empty byte string
        frames.append(frame)
        frame = f.readframes(1)
    for n_sample in range(0, len(frames), size):
        sample_base = []
        for sample in frames[n_sample:(n_sample+size)]:
            i = int.from_bytes(sample, byteorder='little', signed=True)
            i = i / 32767 # turns the amplitude into a float btw -1.0 and 1.0
            i = i + 1
            i = i / 2
            # This transformation is reversed in the postprocessor
            sample_base.append(i)
        res.append(sample_base)
    res = np.array(res)
    for sample in res:
        np.fft.fft(sample) # fourier transform
    return res
def transcodeFile(filename):
    fname, fextension = os.path.splitext(filename)
    outname = fname + '-conv.wav'
    if(os.path.isfile(outname)):
        return outname
    with open(os.devnull, 'w') as devnull:
        #, stdout=devnull, stderr=devnull)
        subprocess.run(['sox', filename, outname, 'channels', '1'])
    return outname

# Converts the first size samples of a wave file to a numpy array 
def wavToNumpy(filename, size):
    outname = transcodeFile(filename)
    with wave.open(outname) as f:
        return fileToNpArray(f, size)
