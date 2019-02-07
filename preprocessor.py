import wave
import subprocess
import numpy as np
import os

def fileToNpArray(f):
    size = 30*f.getframerate()
    base = []
    print(size)
    for n in range(0,size):
        i = int.from_bytes(f.readframes(1), byteorder='little', signed=True)
        base.append(i)
    return np.array(base)

def transcodeFile(filename):
    f = wave.open(filename, 'rb')
    fname, fextension = os.path.splitext(filename)
    outname = fname + '-conv.wav'
    f.close()
    with open(os.devnull, 'w') as devnull:
        subprocess.run(['sox', filename, outname, 'channels', '1', ])#, stdout=devnull, stderr=devnull)
    return outname

def wavToNumpy(filename):
    outname = transcodeFile(filename)
    with wave.open(outname) as f:
        return fileToNpArray(f)
