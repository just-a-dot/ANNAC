import preprocessor
import postprocessor
import sys
import os

import numpy as np

from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import print_summary

from random import shuffle
from multiprocessing import Pool

if len(sys.argv) < 2 and not os.path.isfile('x-train.npy'):
    print("Usage: main.py followed by a list of soundfiles")
    exit()

# BASIC PARAMETERS
sample_rate = 22050
# Size of the samples
# (decides network size, ram usage increases exponentially with this)
size = round(sample_rate * 0.1) 

compression_rate = 0.1
comp_size = round(size*compression_rate)

# NETWORK ARCHITECTURE
input_layer = Input(shape=(size,))
    
encoded = Dense(comp_size*2, activation='sigmoid')(input_layer) #relu
encoded = Dense(comp_size, activation='sigmoid')(encoded)
 
decoded = Dense(size, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)
encoded_input = Input(shape=(comp_size,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

if os.path.isfile('model_weights.h5'):
    print('Saved weights model_weights.h5 found! Attempting to load')
    autoencoder.load_weights('model_weights.h5')
    print('Succesfully loaded weights')

print_summary(autoencoder)
autoencoder.compile(optimizer='adam', loss='mse') 

# PREPROCESSING
wavs = []   # list of wave files CONVERTED TO NUMPY ARRAYS
names = []  # list of filenames
i = 1
for arg in sys.argv[1:]:
    names.append(arg)
def preprocess_par(song):
    return preprocessor.wavToNumpy(song, size)
with Pool() as p:
    print("Loading files. THIS MAY TAKE A VERY LONG TIME")
    wavs = p.map(preprocess_par, sys.argv[1:])
    print(len(wavs))

# Attempt to load previously saved training data
if os.path.isfile('x_train.npy') and os.path.isfile('x_test.npy'):
    print('''Found saved traning data!
          Using given arguments for processing, not training.''')
    with open('x_train.npy', 'rb') as f:
        x_train = np.load(f)
        print(x_train.shape)
    with open('x_test.npy', 'rb') as f:
        x_test = np.load(f)
        print(x_test.shape)
else:
    print("Prepating training data")
    wavs_trans = []
    for song in wavs:
        for sample in song:
            if len(sample) != size:
                continue
            wavs_trans.append(sample)
    shuffle(wavs_trans)
    # The training set is composed of 90% of samples,
    # the test set of the remaining 10%
    x_train = np.asarray(wavs_trans[:(round(0.9*len(wavs_trans)))])
    x_test = np.asarray(wavs_trans[(round(0.9*len(wavs_trans))):])
    wavs_trans = []
    print('Saving training data')
    with open('x_train.npy', 'wb') as f:
        np.save(f, x_train)
    with open('x_test.npy', 'wb') as f:
        np.save(f, x_test)


# Save the weights after each epoch if they improved
filepath = "weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(
    filepath, 
    monitor='val_loss', 
    verbose=1, 
    save_best_only=True, 
    mode='min')
stopping = EarlyStopping(patience=10)
callback_list = [checkpoint, stopping]
#Training happens here

autoencoder.fit(
    x_train, 
    x_train, 
    epochs=3000, 
    batch_size=50, 
    validation_data=(x_test, x_test), 
    callbacks=callback_list)
autoencoder.save_weights('model_weights.h5')
print('Saving trained model')

# POSTPROCESSING
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

