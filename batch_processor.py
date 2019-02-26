import preprocessor
import sys

import numpy as np

from multiprocessing import Pool
from random import shuffle

def process_all(filenames):
    with Pool() as p:
        res_temp = p.map(process_file, filenames)
        res = np.concatenate(res_temp)
        return res
        

def process_file(filename):
    size = int(sys.argv[1])
    wav = preprocessor.wavToNumpy(filename, size)
    return np.asarray(wav)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: batch_processor SIZE")
        print("filenames are read from stdin")
        exit()
    temp_names = []
    for line in sys.stdin:
        temp_names.append(line.rstrip())
    wavs = process_all(temp_names)
    shuffle(wavs)
    x_train = wavs[:(round(0.9*len(wavs)))]
    x_test = wavs[(round(0.9*len(wavs))):]
    print(x_train.shape)
    print(x_test.shape)
    with open('x_train.npy', 'wb') as f:
        np.save(f, x_train)
    with open('x_test.npy', 'wb') as f:
        np.save(f, x_test)
    
