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
    if directory[-1] == '/':
        directory = directory[:-1]
    return glob.glob(directory + '/**/*' + extension, recursive=True)


def process(model_folder, audio_folder, audio_file):
    encoder = load_model(model_folder + '/encoder.hdf5')
    print('Finished loading encoder')
    decoder = load_model(model_folder + '/decoder.hdf5')
    print('Finished loading decoder')

    _, sample_size = encoder.layers[0].output_shape

    audio_files = []
    if audio_file is not None:
        audio_files.append(audio_file)
    else:
        audio_files = get_all_files_in_directory(audio_folder, '.au')

    audio_data = Pool().map(partial(preprocessor.wav_to_numpy, sample_size=sample_size), audio_files)
    print('Finished loading audio files.')
    
    for wav_data, filename in zip(audio_data, audio_files):
        song_data = []
        print('Processing song ' + filename)
        for sample in wav_data:
            encoded_sample = encoder.predict(np.expand_dims(sample, axis=0), batch_size=1)
            decoded_sample = decoder.predict(encoded_sample)
            song_data.append(decoded_sample)
        postprocessor.numpyToWav(song_data, filename[:-3] + '-out.wav')
    

def print_usage_and_exit():
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