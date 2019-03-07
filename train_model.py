import sys
import getopt
from multiprocessing import Pool
from functools import partial
import numpy as np
import os
import glob

from keras.utils import print_summary
from keras.callbacks import ModelCheckpoint

import preprocessor

def train(module_name, npy_file, training_data_path, output_npy, model_output):
    model_module = __import__(module_name, fromlist=[''])
    model = model_module.AEModel()
    
    autoencoder = model.get_autoencoder()
    print_summary(autoencoder)
    autoencoder.compile(optimizer=model.get_optimizer(), loss=model.get_loss_function())

    sample_size = model.get_sample_size()
    audio_data = None

    if (npy_file is not None):
        print('\n\n\nUsing previously saved npy-file.')
        audio_data = load_npy_file(npy_file)
        print('finished loading npy-file.')
    else:
        print('\n\n\nCreating npy from audio files.')
        
        audio_files = get_all_files_in_directory(training_data_path, '.au')

        # handle all files in parallel
        audio_data = Pool().map(partial(preprocessor.wav_to_numpy, sample_size=sample_size), audio_files)
        audio_data = np.array([j for i in audio_data for j in i])

        print('finished loading audio files.')

        if output_npy is not None:
            print('Saving npy array to file ' + output_npy + '.')
            with open(output_npy, 'wb') as f:
                np.save(f, audio_data)

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

    training_data, test_data = prepare_npy_data(audio_data)

    autoencoder.fit(
        training_data, 
        training_data, 
        epochs=model.get_epochs(),
        shuffle=True,
        batch_size=model.get_batch_size(), 
        validation_data=(test_data, test_data), 
        callbacks=callback_list)
    
    weight_files = get_all_files_in_directory(improvement_dir, '.hdf5')
    latest_improvement = max(weight_files, key=os.path.getctime)

    #load best weights and save model
    autoencoder.load_weights(latest_improvement)
    try:
        os.makedirs(model_output)
    except FileExistsError:
        # directory already exists
        pass
    
    autoencoder.save(model_output + '/autoencoder.hdf5')
    model.get_encoder().save(model_output + '/encoder.hdf5')
    model.get_decoder().save(model_output + '/decoder.hdf5')
    

def get_all_files_in_directory(directory, extension=''):
    if directory[-1] == '/':
        directory = directory[:-1]
    return glob.glob(directory + '/**/*' + extension, recursive=True)

def load_npy_file(npy_file):
    with open(npy_file, 'rb') as f:
        return np.load(f)

def prepare_npy_data(npy_data):
    np.random.shuffle(npy_data)

    size, _ = npy_data.shape
    
    npy_data = np.split(npy_data, [round(size*0.9), size])

    training_data = npy_data[0]
    test_data = npy_data[1]

    return training_data, test_data

def print_usage_and_exit():
    print('Usage: python annac.py [-m/--module-name=] <module-containing-the-model> [-s/--save-model=] <where-to-save-the-model (dir)>\n\
        [-n/--npy-file=] <training-data.npy> OR \n\
        [-t/--training-data=] <folder-with-audio-data (extension:au/wav)> [-o/--output-npy=] <where-to-output-npy-file>')
    sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print_usage_and_exit()

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'm:n:t:o:s:', ['module-name=', 'npy-file=', 'training-data=', 'output-npy=', '--save-model-to='])
    except getopt.GetoptError as e:
        print(e)
        print_usage_and_exit()
        
    module_name = None
    npy_file = None
    training_data = None
    output_npy = None
    model_output = None

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

    train(module_name, npy_file, training_data, output_npy, model_output)