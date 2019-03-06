import preprocessor
import postprocessor
import sys
import os

import numpy as np

import keras
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import print_summary

from random import shuffle

if len(sys.argv) < 3 and not os.path.isfile('x-train.npy'):
    print("Usage: main.py file/folder/reload/retrain followed by a list of soundfiles")
    exit()

##################################
#       network params
##################################

sample_rate = 22050
# Size of the samples
# (decides network size, ram usage increases exponentially with this)
size = round(sample_rate * 0.1) 

compression_rate = 0.1
comp_size = round(compression_rate * size)


#learning_rate = 0.00001
learning_rate = 0.00001
optimizer = keras.optimizers.Adam(lr=learning_rate)
loss = 'mse'


epochs = 1000
batch_size = 75

##################################

# NETWORK ARCHITECTURE
input_layer = Input(shape=(size,))
    
encoded = Dense(comp_size*8, activation='sigmoid')(input_layer) #reluc
encoded = Dense(comp_size*5, activation='sigmoid')(encoded)
encoded = Dense(comp_size*2, activation='sigmoid')(encoded)
encoded = Dense(comp_size, activation='sigmoid')(encoded)

 
decoded = Dense(size, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)
encoded_input = Input(shape=(comp_size,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))


print_summary(autoencoder)
autoencoder.compile(optimizer=optimizer, loss=loss)


# PREPROCESSING
wavs = []   # list of wave files CONVERTED TO NUMPY ARRAYS
names = []  # list of filenames
i = 1
soundfiles = []
if sys.argv[1] == 'file' or sys.argv[1] == 'process_file':
    soundfiles = sys.argv[2:]
    for arg in soundfiles:
        print("Reading file " 
            + str(i) + "/" 
            + str(len(sys.argv[2:])) 
            + "              ", end='\r')
        wavs.append(preprocessor.wavToNumpy(arg, size))
        names.append(arg)
        i += 1

if sys.argv[1] == 'folder':
    for p in sys.argv[2:]:        
        for root, dirs, files in os.walk(os.path.abspath(p)):
            for filename in files:
                fname, fextension = os.path.splitext(filename)
                if fextension != ".au":
                    continue
                else:                
                    soundfiles.append(os.path.join(root, filename))





# Attempt to load previously saved training data
if sys.argv[1] == 'process_file':
    pass
    # do nothing so we don't fit
elif os.path.isdir('np_train') and os.path.isdir('np_test') and sys.argv[1] != "reload" and sys.argv[1] != 'folder':
    print('''Found saved training data!
        Using given arguments for processing, not training.
        Loading...''')
    train_data = []
    test_data = []
    for root, dirs, files in os.walk(os.path.abspath('np_train')):
        for filename in files:
            with open('np_train/' + filename, 'rb') as f:
                train_data.append(np.load(f))

    if len(train_data) > 1:
        x_train = np.concatenate(tuple(train_data))
    else:
        x_train = train_data

    for root, dirs, files in os.walk(os.path.abspath('np_test')):
        for filename in files:
            with open('np_test/' + filename, 'rb') as f:
                test_data.append(np.load(f))

    if len(test_data) > 1:
        x_test = np.concatenate(tuple(test_data))
    
else:
    os.makedirs('np_train')
    os.makedirs('np_test')
    step_size = 100
    ii = 0
    k = 0
    for i in range(0, len(soundfiles), step_size):
        current_folder = str(sys.argv[2+ii]).rsplit('/')[-1]
        ii += 1

        current_data = []
        output_data = []

        for arg in soundfiles[i:i+step_size]:
            k += 1
            print("Reading file " 
                + str(k) + "/" 
                + str(len(soundfiles)) 
                + "              ", end='\r')

            npdata = preprocessor.wavToNumpy(arg, size)
            current_data.append(npdata)
        for song in current_data:
            for sample in song:
                if len(sample) != size:
                    continue
                output_data.append(sample)
        shuffle(output_data)

        # The training set is composed of 90% of samples,
        # the test set of the remaining 10%
        x_train = np.asarray(output_data[:(round(0.9*len(output_data)))])
        x_test = np.asarray(output_data[(round(0.9*len(output_data))):])

        print('Saving training data')
        with open('np_train/' + current_folder + '.npy', 'wb') as f:
            np.save(f, x_train)
        with open('np_test/' + current_folder + '.npy', 'wb') as f:
            np.save(f, x_test)



if os.path.isfile('model_weights.h5'):
    print('Saved weights model_weights.h5 found! Attempting to load')
    autoencoder.load_weights('model_weights.h5')
    print('Successfully loaded weights')

# Save the weights after each epoch if they improved
filepath = "weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(
    filepath, 
    monitor='val_loss', 
    verbose=1, 
    save_best_only=True, 
    mode='min')
callback_list = [checkpoint]
#Training happens here

if sys.argv[1] == "process_file" and os.path.isfile('model_weights.h5'):
    print('model already trained, skipping the fitting part, just output.')

    if len(wavs) == 0:
        print('No files to process given, terminating')
        exit(0)
    # Reassemble wavs
    for i in range(0, len(wavs)):
        base = []
        print("Processing song " + str(i+1) + " of " + str(len(wavs)))
        for sample in wavs[i]:
            if len(sample) != size:
                continue
            encoded_wav = encoder.predict(
                np.expand_dims(sample, axis=0), 
                batch_size=1)
            decoded_wav = decoder.predict(encoded_wav)
            base.append(decoded_wav)
            #base.append(np.expand_dims(sample, axis=0))
        postprocessor.numpyToWav(base, names[i] + '-out.wav')

else:
    np.random.shuffle(x_train)
    np.random.shuffle(x_test)

    for i in range(epochs):
        autoencoder.fit(
            x_train, 
            x_train, 
            epochs=1000*(i+1),
            shuffle=True,
            initial_epoch=i*1000,
            batch_size=batch_size, 
            validation_data=(x_test, x_test), 
            callbacks=callback_list)
        autoencoder.save_weights('model_weights_iteration_' + str(i) + '.h5')
        print('Saving trained model')
        # POSTPROCESSING
        if len(wavs) == 0:
            print('No files to process given, terminating')
            exit(0)
        # Reassemble wavs
        for k in range(0, len(wavs)):
            base = []
            print("Processing song " + str(k+1) + " of " + str(len(wavs)))
            for sample in wavs[k]:
                if len(sample) != size:
                    continue
                encoded_wav = encoder.predict(
                    np.expand_dims(sample, axis=0), 
                    batch_size=1)
                decoded_wav = decoder.predict(encoded_wav)
                base.append(decoded_wav)
                #base.append(np.expand_dims(sample, axis=0))
            postprocessor.numpyToWav(base, names[k] + '-' + str(i) + '-out.wav')

