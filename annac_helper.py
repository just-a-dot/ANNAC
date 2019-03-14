import glob
from random import shuffle
from math import ceil
from multiprocessing import Pool
from functools import partial
import numpy as np
import preprocessor

def get_all_files_in_directory(directory, extension=''):
    '''
    A function used to recursivly extract all files with the given extension from a directory.

    :param directory: The directory we want to extract the files from.
    :param extension: The extension we want to use. All files are retrieved on default.

    :returns: The files in the directory with the given extension.
    '''
    if directory[-1] == '/':
        directory = directory[:-1]
    return glob.glob(directory + '/**/*' + extension, recursive=True)

def get_song_number_for_filename(filename):
    '''
    A function return the song number for the given filename.
    Format normally is <genre>.xxxxx.au
    '''
    split_name = filename.split('.')
    if len(split_name) == 3:
        return int(split_name[1])
    else:
        return -1


def get_audio_data_for_lstm(audio_files, input_size, song_length_in_frames, chunk_size):
    '''
    A function to prepare the audio files in a way suitable for lstm networks.

    :param audio_files: The files we want to get the data from
    :param input_size: The input size of the network
    :param song_length_in_frames: The max length of a song we want to process (for uniformity)
    :param chunk_size: The size of a chunk for training the network.

    :returns: The prepared audio data.
    '''
    audio_data = Pool().map(partial(preprocessor.wav_to_numpy, input_size=input_size, song_length_in_frames=song_length_in_frames), audio_files)
    
    sequence = np.zeros((ceil(song_length_in_frames/chunk_size)*len(audio_data), chunk_size, input_size))

    song_i = 0
    for song in audio_data:
        sample_i = 0
        chunk_i = 0
        for sample in song:
            if(len(sample) == input_size):
                sequence[(song_i*(song_length_in_frames//chunk_size)) + chunk_i][sample_i] = sample
                sample_i += 1
            if sample_i == chunk_size:
                chunk_i += 1
                sample_i = 0
        song_i += 1

    return sequence

def get_audio_data_normally(audio_files, input_size):
        '''
    A function to prepare the audio files in a way suitable for "normal" networks.

    :param audio_files: The files we want to get the data from
    :param input_size: The input size of the network

    :returns: The prepared audio data.
    '''
    audio_data = Pool().map(partial(preprocessor.wav_to_numpy, input_size=input_size), audio_files)
    return np.array([j for i in audio_data for j in i])