import sys
import getopt
from multiprocessing import Pool
from functools import partial
import numpy as np
import os
import signal
import glob

from keras.utils import print_summary
from keras.callbacks import ModelCheckpoint

import preprocessor

autoencoder = None
encoder = None
decoder = None
improvement_dir = None

def train(module_name, npy_file, training_data_path, output_npy, model_output):
    ''' 
    A function to train a model as specified in the module name with the given training data.
    We either train using a numpy file or a folder with all the training data.

    :param module_name: The name of the module we attempt to load in order to get the model.
    :param npy_file: A numpy file with the training data. Can be None.
    :param training_data_path: A path to the training data. Can be None
    :param output_npy: If we use a folder as training data, we can also save the output to a numpy file.
    :param model_output: The folder in which we want to save the model.
    '''
    # global vars bc we need those later if we want to save upon ctrl+c
    global autoencoder, encoder, decoder, improvement_dir
    
    # import the module containing the model.
    model_module = __import__(module_name, fromlist=[''])
    model = model_module.AEModel()

    # get the models    
    autoencoder = model.get_autoencoder()
    encoder = model.get_encoder()
    decoder = model.get_decoder()

    print_summary(autoencoder)
    autoencoder.compile(optimizer=model.get_optimizer(), loss=model.get_loss_function())

    input_size = model.get_input_size()

    audio_data = get_training_data(npy_file, training_data_path, output_npy)

    # where to save the weight improvements during training
    improvement_dir = 'weights_improvement/' + module_name
    try:
        os.makedirs(improvement_dir)
    except FileExistsError:
        # directory already exists
        pass

    improvement_file_format = improvement_dir + '/{epoch:02d}-{val_loss:.10f}.hdf5'
    
    checkpoint = ModelCheckpoint(
        improvement_file_format, 
        monitor='val_loss', 
        verbose=1, 
        save_best_only=True, 
        mode='min')
    callback_list = [checkpoint]

    # split training data 
    training_data, test_data = prepare_npy_data(audio_data)

    # register the ctrl+c signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # train
    autoencoder.fit(
        training_data, 
        training_data, 
        epochs=model.get_epochs(),
        shuffle=True,
        batch_size=model.get_batch_size(), 
        validation_data=(test_data, test_data), 
        callbacks=callback_list)
    
    save_all_models()
    

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

def get_training_data(npy_file, training_data_path, output_npy):
    '''
    A function to get the training data from either a npy file or a folder of files.

    :param npy_file: The path to the numpy file containing the data.
    :param training_data_path: The path to the folder containing the training data
    :param output_npy: Where to save the processed data (if a folder was used)

    :returns: The audio data used for training the model.
    '''
    
    audio_data = None
    if (npy_file is not None):
        print('\n\n\nUsing previously saved npy-file.')
        audio_data = load_npy_file(npy_file)
        print('finished loading npy-file.')
    else:
        print('\n\n\nCreating npy from audio files.')
        
        audio_files = get_all_files_in_directory(training_data_path, '.au')
        # ignore the last 10% of the audio files for later (proper) validation
        audio_files = list(filter(lambda x: get_song_number_for_filename(x) < 90, audio_files))

        # convert files to wav and get the numpy representation in parallel
        audio_data = Pool().map(partial(preprocessor.wav_to_numpy, input_size=input_size), audio_files)
        audio_data = np.array([j for i in audio_data for j in i])

        print('finished loading audio files.')

        if output_npy is not None:
            # save the processed data to numpy file if specified
            print('Saving npy array to file ' + output_npy + '.')
            with open(output_npy, 'wb') as f:
                np.save(f, audio_data)
    return audio_data

def get_all_files_in_directory(directory, extension=''):
    '''
    A function used to recursivly extract all files with the given extension from a directory.

    :param directory: The directory we want to extract the files from.
    :param extension: The extension we want to use. All files are retrieved on default.

    :returns: The files in the directory with the given extension.
    '''
    # cut the last '/' from the directory
    if directory[-1] == '/':
        directory = directory[:-1]
    return glob.glob(directory + '/**/*' + extension, recursive=True)

def load_npy_file(npy_file):
    '''
    A function to load a file as a numpy array.

    :param npy_file: The file we want to load.

    :returns: The loaded numpy array.
    '''
    with open(npy_file, 'rb') as f:
        return np.load(f)

def prepare_npy_data(npy_data):
    '''
    A function to split and shuffle the data in order to make it useful for training.
    We use 90% for training and 10% for validation.

    :param npy_data: The data we want to prepare

    :returns: A tuple of training_data,test_data. 
    '''
    np.random.shuffle(npy_data)

    size, _ = npy_data.shape
    
    npy_data = np.split(npy_data, [round(size*0.9), size])

    return npy_data[0], npy_data[1]

def print_usage_and_exit():
    '''
    A function to print the usage of the script and then exit.
    '''
    print('Usage: python annac.py [-m/--module-name=] <module-containing-the-model> [-s/--save-model=] <where-to-save-the-model (dir)>\n\\
        [-n/--npy-file=] <training-data.npy> OR \n\
        [-t/--training-data=] <folder-with-audio-data (extension:au/wav)> [-o/--output-npy=] <where-to-output-npy-file>')
    sys.exit(1)

def save_all_models():
    '''
    A function to save all models. The function has no parameters since it used global variables.
    These have to be global in case this function is being called from the ctrl+c signal handler.
    '''

    global autoencoder, decoder, encoder, improvement_dir, model_output

    print('Saving all models')

    # The model might have trained further and worsened, therefore we need to load the best weights now.
    weight_files = get_all_files_in_directory(improvement_dir, '.hdf5')
    latest_improvement = max(weight_files, key=os.path.getctime)

    autoencoder.load_weights(latest_improvement)
    try:
        os.makedirs(model_output)
    except FileExistsError:
        # directory already exists
        pass
    
    autoencoder.save(model_output + '/autoencoder.hdf5')
    encoder.save(model_output + '/encoder.hdf5')
    decoder.save(model_output + '/decoder.hdf5')

def signal_handler(sig, frame):
    '''
    A signal handler that will save all models and then exit gracefully.

    :param sig: Not used, here for compatability.
    :param frame: Not used, here for compatability.
    '''
    save_all_models()
    sys.exit(0)


if __name__ == "__main__":   
    if len(sys.argv) < 4:
        print_usage_and_exit()

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'm:n:t:o:s:', ['module-name=', 'npy-file=', 'training-data=', 'output-npy=', '--save-model-to='])
    except getopt.GetoptError as e:
        # script wasnt called properly, tell the user and exit.
        print(e)
        print_usage_and_exit()
        
    module_name = None
    npy_file = None
    training_data = None
    output_npy = None
    model_output = None
    model_input = None

    # extract arguments
    for opt, arg in opts:
        if opt in ('-m', '--module-name'):
            module_name = arg
        elif opt in ('-n', '--npy-file'):
            npy_file = arg
        elif opt in ('-t', '--training-data'):
            training_data = arg
        elif opt in ('-o', '--output-npy'):
            output_npy = arg
        elif opt in ('-s', '--save-model-to'):
            model_output = arg
        elif opt in ('-c', '--continue'):
            model_input = arg
        else:
            print_usage_and_exit()

    if model_output is None:
        print('You have to specify where to save the model to.')
        print_usage_and_exit()                
    
    if training_data is not None and npy_file is not None:
        print('Use either --training-data or --npy-file, not both!')
        print_usage_and_exit()
    elif module_name is None:
        print('Specify the module_name where the model can be found.')
        print_usage_and_exit()

    # run the training
    train(module_name, npy_file, training_data, output_npy, model_output)