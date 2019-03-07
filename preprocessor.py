import wave
import subprocess
import numpy as np
import os

def file_to_np_array(audio_file, sample_size):
    frames = []
    res = []
    frame = audio_file.readframes(1)
    while frame != b'': # b'' = empty byte string
        frames.append(frame)
        frame = audio_file.readframes(1)

        if len(frames) % sample_size == 0:
            samples = []
            for sample in frames:
                i = int.from_bytes(sample, byteorder='little', signed=True)
                i = i / 32767 # turns the amplitude into a float btw -1.0 and 1.0
                i = i + 1
                i = i / 2
                # This transformation is reversed in the postprocessor
                samples.append(i)
            res.append(samples)

            # clear frames
            frames = []

    return np.array(res)

def convert_file_to_wav(filename):
    fname, _ = os.path.splitext(filename)
    wav_filename = fname + '.wav'

    if(os.path.isfile(wav_filename)):
        return wav_filename
  
    subprocess.run(['sox', filename, wav_filename, 'channels', '1'])

    return wav_filename

# Converts the first sample_size samples of a wave file to a numpy array 
def wav_to_numpy(filename, sample_size):
    wav_filename = convert_file_to_wav(filename)
    with wave.open(wav_filename) as wav_file:
        return file_to_np_array(wav_file, sample_size)