import wave
import subprocess
import numpy as np
import os

def file_to_np_array(audio_file, input_size, song_length_in_frames):
    '''
    A function to turn a wave file into a numpy array.
    If the last sample of the file is not as large as input_size, it is ignored.

    :param audio_file: The path to the file we want to transform.
    :param input_size: How many frames form a sample.

    :returns: The wave file as numpy array (in regard to the sample size)
    '''
    frames = []
    res = []

    # get the first frame
    frame = audio_file.readframes(1)
    frame_counter = 1

    # while file not done
    while frame != b'': # b'' = empty byte string
        frames.append(frame)
        frame = audio_file.readframes(1)
        frame_counter += 1

        # we've reached sample size, so turn the previous frames into one sample
        # and clear our the frames array.
        if len(frames) % input_size == 0:
            samples = []
            for sample in frames:
                # transform the wave file into something the model can handle better
                i = int.from_bytes(sample, byteorder='little', signed=True)
                i = i / 32767 # turns the amplitude into a float btw -1.0 and 1.0
                i = i + 1 # range now: [0..2]
                i = i / 2 # range now: [0..1]
                samples.append(i)
            res.append(samples)
            frames = []

        if song_length_in_frames is not None and frame_counter > song_length_in_frames:
            break

    return np.array(res)

def convert_file_to_wav(filename):
    '''
    A function to convert a file into wave.
    The command line tool "sox" is used for this.
    If the file already exists, just the path is returned.

    :param filename: The file we want to convert.

    :returns: The filename of the new wave file.
    '''
    fname, _ = os.path.splitext(filename)
    wav_filename = fname + '.wav'

    if(os.path.isfile(wav_filename)):
        return wav_filename
  
    subprocess.run(['sox', filename, wav_filename, 'channels', '1'])

    return wav_filename

def wav_to_numpy(filename, input_size, song_length_in_frames=None):
    '''
    A function to convert a file into wav and then into a numpy array.

    :param filename: The file we want to get the data for
    :param input_size: How many frames should form one sample. Correlates to the input size of the model.

    :returns: A numpy array containing the data.
    '''
    wav_filename = convert_file_to_wav(filename)
    with wave.open(wav_filename) as wav_file:
        return file_to_np_array(wav_file, input_size, song_length_in_frames)