import sys
import getopt

from keras.models import load_model

import preprocessor
import annac_helper

def validate(model_folder, audio_folder):
    '''
    A function to calculate the metrics of a trained network with the given dataset.
    We will only use the audio files with index >89, since those should have been discarded during training.
    Therefore, the network should not have seen those files yet.

    :param model_folder: Where the model files reside
    :param audio_folder: Where the audio files reside
    '''

    autoencoder = load_model(model_folder + '/autoencoder.hdf5')

    audio_files = annac_helper.get_all_files_in_directory(audio_folder, extension='.au')
    audio_files = list(filter(lambda x: annac_helper.get_song_number_for_filename(x) < 90, audio_files))
    
    _, input_size = autoencoder.layers[0].output_shape
    audio_data = Pool().map(partial(preprocessor.wav_to_numpy, input_size=input_size), audio_files)
    print('Finished loading audio files.')

    return autoencoder.evaluate(x=audio_data, y=audio_data, batch_size=75, )


def print_usage_and_exit():
    '''
    A function to print the usage of the script and then exit.
    '''
    print('Usage: python process_files [-m/--model-folder=] <keras-model-folder> \n\
        [-a/--audio-folder] <folder-with-audiofiles>')
    sys.exit(1)

if __name__ == "__main__":
    if (len(sys.argv) < 3):
        print_usage_and_exit()
    else:
        try:
            opts, args = getopt.getopt(sys.argv[1:], 'm:a:', ['model-folder=', 'audio-folder='])
        except getopt.GetoptError as e:
            print(e)
            print_usage_and_exit()


        model_folder = None
        audio_folder = None

        # extract arguments
        for opt, arg in opts:
            if opt in ('-a', '--audio-folder'):
                audio_folder = arg
            elif opt in ('-m', '--model-folder'):
                model_folder = arg
            else:
                print_usage_and_exit()

        if model_folder is None:
            print('please specify the model folder')
            print_usage_and_exit()
        elif audio_folder is None:
            print('please specify the audio folder')
            print_usage_and_exit()
        
        print(validate(model_folder, audio_folder))