import wave
import subprocess
import numpy as np
import os

def fileToNpArray(f, size, song_length):
    res = []
    frames = []
    frame = f.readframes(1)
    while frame != b'':
        frames.append(frame)
        frame = f.readframes(1)
    for n_sample in range(0, len(frames), size):
        sample_base = []
        if n_sample >= song_length:
            break
        for sample in frames[n_sample:(n_sample+size)]:
            i = int.from_bytes(sample, byteorder='little', signed=True)
            i = i / 32767 #Turn the amplitude into a float between -1.0 and 1.0
            i = i + 1
            i = i / 2
            # This transformation is reversed in the postprocessor
            sample_base.append(i)
        res.append(np.fft.fft(sample_base).real)
        #res.append((sample_base))
    return np.array(res)

def transcodeFile(filename):
    fname, fextension = os.path.splitext(filename)
    outname = fname + '-conv.wav'
    with open(os.devnull, 'w') as devnull:
        subprocess.run(['sox', filename, outname, 'channels', '1'])#, stdout=devnull, stderr=devnull)
    return outname

def wavToNumpy(filename, size, song_length):
    outname = transcodeFile(filename)
    with wave.open(outname) as f:
        return fileToNpArray(f, size, song_length)
