import sys
import getopt
import importlib
import glob
from multiprocessing import Pool
from functools import partial
import numpy as np

from keras.models import load_model
import preprocessor
import postprocessor

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


def process(model_folder, audio_folder, audio_file):
    '''
    A function to to process one or more audio files.
    The file will be compressed and then decompressed. The output will be saved.

    :param model_folder: The folder where the trained models reside.
    :param audio_folder: A folder containing the audio files to be processed.
    :param audio_file: A single audio file to be processed.
    '''

    encoder = load_model(model_folder + '/encoder.hdf5')
    print('Finished loading encoder')
    decoder = load_model(model_folder + '/decoder.hdf5')
    print('Finished loading decoder')

    # get the input size of the model in order to preprocess the audio files properly.
    _, input_size = encoder.layers[0].output_shape

    audio_files = []
    if audio_file is not None:
        audio_files.append(audio_file)
    else:
        audio_files = get_all_files_in_directory(audio_folder, '.au')

    audio_data = Pool().map(partial(preprocessor.wav_to_numpy, input_size=input_size), audio_files)
    print('Finished loading audio files.')
    
    # process the songs.
    for wav_data, filename in zip(audio_data, audio_files):
        song_data = []
        print('Processing song ' + filename)
        for chunk in wav_data:
            encoded_chunk = encoder.predict(np.expand_dims(chunk, axis=0), batch_size=1)
            decoded_chunk = decoder.predict(encoded_chunk)
            song_data.append(decoded_chunk)
        postprocessor.numpyToWav(song_data, filename[:-3] + '-out.wav')
    

def print_usage_and_exit():
    '''
    A function to print the usage of the script and then exit.
    '''
    print('Usage: python process_files [-m/--model-folder=] <keras-model-folder> \n\
        [-d/--directory=] <folder-with-audiofiles> OR \n \
        [-f/--file=] <single-audiofile>')
    sys.exit(1)

if __name__ == "__main__":
    if (len(sys.argv) < 3):
        print_usage_and_exit()
    else:
        try:
            opts, args = getopt.getopt(sys.argv[1:], 'm:d:f:', ['model-folder=', 'directory=', 'file='])
        except getopt.GetoptError as e:
            print(e)
            print_usage_and_exit()


        model_file = None
        audio_folder = None
        audio_file = None

        # extract arguments
        for opt, arg in opts:
            if opt in ('-d', '--directory'):
                audio_folder = arg
            elif opt in ('-m', '--model-folder'):
                model_file = arg
            elif opt in ('-f', '--file'):
                audio_file = arg
            else:
                print_usage_and_exit()

        if model_file is None:
            print('please specify the model folder')
            print_usage_and_exit()
        elif audio_file is not None and audio_folder is not None:
            print('Use either --file or --directory, not both!')
            print_usage_and_exit()
        
        process(model_file, audio_folder, audio_file)